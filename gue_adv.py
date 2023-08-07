import argparse
import copy
import math
import numpy as np
import os
import time
from tqdm import tqdm
from pprint import pprint
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import ResNet18, VGG, UNet
from attacks.adv import batch_adv_delta, batch_adv_attack
from attacks.trades import batch_trades_delta, batch_trades_attack
from utils import setup_seed

def create_net(args):
    if args.dataset == "cifar10":
            return ResNet18(num_classes=args.num_classes)
    if args.dataset == "cifar100":
            return ResNet18(num_classes=args.num_classes)

def f(args, clsmodel, x, y):
    logits = clsmodel(x)
    losses = []
    for i in range(1, args.num_classes):
        k = (y+i) % args.num_classes
        losses.append(nn.CrossEntropyLoss(reduction='none')(logits, k))
    losses = torch.stack(losses, dim=0)
    losses = losses.max(0)[0].mean()
    return losses


# gradient of f(theta) over theta
def f_theta(args, clsmodel, x, y, retain_graph=False, create_graph=False):
    loss = f(args, clsmodel, x, y)
    print("clean loss:{:.4f}".format(loss.item()))
    grad = torch.autograd.grad(loss, clsmodel.parameters(),
                               retain_graph=retain_graph,
                               create_graph=create_graph)
    return grad, loss


def g(args, atkmodel, clsmodel, x, y):
    noise = atkmodel(x) * args.eps
    atkdata = torch.clamp(x + noise, 0, 1)
    delta = batch_trades_delta(args, clsmodel, atkdata, y)
    logits = clsmodel(torch.cat((atkdata, atkdata+delta), dim=0))
    logits_cln, logits_adv = logits[:logits.size(0) // 2], logits[logits.size(0) // 2:]
    kl = nn.KLDivLoss(reduction='batchmean')
    loss_rob = kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_cln, dim=1))
    loss_nat = nn.CrossEntropyLoss()(logits_cln, y)
    loss = loss_nat + args.beta * loss_rob
    # delta = batch_adv_delta(args, clsmodel, x, y)
    # logits = clsmodel(atkdata +  delta)
    # loss = nn.CrossEntropyLoss()(logits, y)
    return loss


# gradient on theta of g(a, theta)
def g_theta(args, atkmodel, clsmodel, x, y, retain_graph=False, create_graph=False):
    loss = g(args, atkmodel, clsmodel, x, y)
    print("poison loss:{:.4f}".format(loss.item()))
    grad = torch.autograd.grad(loss, clsmodel.parameters(),
                               retain_graph=retain_graph,
                               create_graph=create_graph)
    return grad


# gradient on a of g(a, theta)
def g_v(args, atkmodel, clsmodel, x, y, retain_graph=False, create_graph=False):
    loss = g(args, atkmodel, clsmodel, x, y)
    grad = torch.autograd.grad(loss, atkmodel.parameters(),
                               retain_graph=retain_graph,
                               create_graph=create_graph)
    return grad


# compute q^(a, theta^T) and gradient over (a, theta)
# tmpmodel: approximation of theta* (T step gradient decent over theta)
def q_a_theta(args, atkmodel, clsmodel, tmpmodel, x, y, retain_graph=False, create_graph=False):
    loss1 = g(args, atkmodel, clsmodel, x, y)
    loss2 = g(args, atkmodel, tmpmodel, x, y)
    loss = loss1 - loss2
    print("poison loss:{:.4f}".format(loss1.item()))
    print("approximate optimal poison loss:{:.4f}".format(loss2.item()))
    grad_q_a = torch.autograd.grad(loss, atkmodel.parameters(),
                                   retain_graph=True,
                                   create_graph=create_graph)
    grad_q_theta = torch.autograd.grad(loss1, clsmodel.parameters(),
                                       retain_graph=retain_graph,
                                       create_graph=create_graph)
    return loss, grad_q_theta, grad_q_a, loss1


def bome(args, atkmodel, clsmodel, clsoptimizer, atkoptimizer, train_loader, iterations, epoch):
    clean_loss_list = []
    poison_loss_list = []
    st_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        print("Epoch:{} batch:{}".format(epoch, batch_idx))
        data, target = data.cuda(), target.cuda()
        tmpmodel = create_net(args).cuda()
        inner_opt = torch.optim.Adam(tmpmodel.parameters(), lr=args.lr_tmp)
        tmpmodel.load_state_dict(clsmodel.state_dict())
        tmpmodel.train()

        for i in range(iterations):
            loss = g(args, atkmodel, tmpmodel, data, target)
            print("iteration:{}, batch loss:{:.4f}".format(i, loss.item()))
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()

        # prepare gradients
        grad_f_theta, clean_loss = f_theta(args, clsmodel, data, target)
        clean_loss_list.append(clean_loss.item())
        grad_f_theta_vec = torch.cat([grad.flatten() for grad in grad_f_theta]).view(-1)
        loss, grad_q_theta, grad_q_a, poison_loss = q_a_theta(args, atkmodel, clsmodel, tmpmodel, data, target)
        poison_loss_list.append(poison_loss.item())
        grad_q_theta_vec = torch.cat([grad.flatten() for grad in grad_q_theta]).view(-1)
        grad_q_a_vec = torch.cat([grad.flatten() for grad in grad_q_a]).view(-1)

        norm_q = (grad_q_theta_vec**2).sum() + (grad_q_a_vec**2).sum()
        norm_q_theta = (grad_q_theta_vec ** 2).sum()
        # the inner product not need to compute grad_g_a_vec since f only contain theta
        vec_product = (grad_f_theta_vec * grad_q_theta_vec).sum()

        print("q:{:.4f}, inner product:{:.4f}, "
              "norm_q:{:.4f}, norm_q_theta:{:.4f}".format(loss.item(), vec_product.item(), norm_q.item(), norm_q_theta.item()))
        lmbd = F.relu((args.eta * norm_q - vec_product) / (norm_q + 1e-8))
        print("lambda:{}".format(lmbd.item()))
        atkmodel.train()
        clsmodel.train()
        atkoptimizer.zero_grad()
        for i, param in enumerate(atkmodel.parameters()):
            param.grad = lmbd * grad_q_a[i]
        atkoptimizer.step()

        clsoptimizer.zero_grad()
        for i, param in enumerate(clsmodel.parameters()):
            param.grad = grad_f_theta[i] + lmbd * grad_q_theta[i]
        clsoptimizer.step()
    end_time = time.time()
    clean_loss = sum(clean_loss_list) / len(clean_loss_list)
    poison_loss = sum(poison_loss_list) / len(poison_loss_list)
    print("time:{}".format(end_time-st_time))
    return clean_loss, poison_loss


def test(args, atkmodel, scratchmodel, train_loader, test_loader,
         epoch, trainepoch):
    args.num_steps = 10
    st_time = time.time()
    test_loss = 0
    correct = 0
    atkmodel.eval()
    testoptimizer = torch.optim.SGD(scratchmodel.parameters(), lr=0.01, momentum=0.9, weight_decay=args.weight_decay)
    for i in range(trainepoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            testoptimizer.zero_grad()
            with torch.no_grad():
                noise = atkmodel(data) * args.eps
                atkdata = torch.clamp(data + noise, 0, 1)
            x_adv = batch_adv_attack(args, scratchmodel, atkdata, target)
            output = scratchmodel(x_adv)
            loss = F.cross_entropy(output, target)
            with torch.no_grad():
                output2 = scratchmodel(data)
                loss2 = F.cross_entropy(output2, target)
            loss.backward()
            testoptimizer.step()
            if batch_idx % args.log_interval == 0:
                print(
                    'Test_train Epoch: {}, {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tNatLoss: {:.6f}'.format(
                        epoch, i, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                        loss.item(),
                        loss2.item()
                    ))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = scratchmodel(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'. \
          format(test_loss, correct, len(test_loader.dataset),
                 100. * correct / len(test_loader.dataset)))
    accorig = 100. * correct / len(test_loader.dataset)
    print('test time:', time.time() - st_time)
    WRITER.add_scalar('acc(origin)', accorig, global_step=epoch - 1)
    WRITER.add_scalar('test loss', test_loss, global_step=epoch - 1)
    return accorig


def make_data_clean(args):
    transform_test = transforms.ToTensor()
    if args.dataset == "cifar10":
        train_set = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_test)
        test_set = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_test)
    elif args.dataset == "cifar100":
        train_set = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_test)
        test_set = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_test)
    elif args.dataset == "SVHN":
        train_set = datasets.SVHN(args.data_path, split="train", transform=transform_test)
        test_set = datasets.SVHN(args.data_path, split="test", transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default="0", type=str)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', "SVHN"])
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--eps', type=float, default=8, help='epsilon for data poison')
    parser.add_argument('--lr_cls', type=float, default=0.01,
                        help='learning rate for classification model')
    parser.add_argument('--lr_tmp', type=float, default=0.0001,
                        help='learning rate for temporal classification model')
    parser.add_argument('--lr_atk', type=float, default=0.1,
                        help='learning rate for attack model')
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument('--eta', type=float, default=1.5, help="super parameters for constraint")
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--iterations', type=int, default=10, help='steps for approximating theta^*')
    parser.add_argument('--epochs', type=int, default=150)

    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--test_epochs', type=int, default=5)
    parser.add_argument('--test-interval', type=int, default=5, metavar='N',
                        help='how many epochs to wait before serious testing')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--tensorboard_path', default='./log', help="tensorboard path")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--eps_at', default=4, type=float)
    parser.add_argument('--constraint', default='Linf', type=str, choices=['Linf', 'L2'])
    parser.add_argument('--beta', type=float, default=1.0,
                        help='regularization, i.e., 1/lambda in TRADES')
    args = parser.parse_args()
    # Attack options
    args.eps_at = args.eps_at / 255
    args.step_size = args.eps_at / 4
    args.num_steps = 10
    args.random_restarts = 1

    args.data_path = os.path.join(args.data_path, args.dataset)
    args.eps = args.eps / 255
    args.lr_milestones = [120, 140]
    args.lr_step = 0.1
    return args


if __name__ == '__main__':
    args = parse_args()
    setup_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    pprint(vars(args))
    WRITER = SummaryWriter(args.tensorboard_path)
    train_loader, test_loader = make_data_clean(args)
    attacker = UNet(3).cuda()
    classifier = create_net(args).cuda()
    cls_optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr_cls)
    cls_schedule = optim.lr_scheduler.MultiStepLR(cls_optimizer, milestones=args.lr_milestones, gamma=args.lr_step)
    atk_optimizer = torch.optim.SGD(attacker.parameters(), lr=args.lr_atk,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    atk_schedule = optim.lr_scheduler.MultiStepLR(atk_optimizer, milestones=args.lr_milestones, gamma=args.lr_step)
    best_acc = 100
    for epoch in range(1, args.epochs + 1):
        clean_loss, poison_loss = bome(args, attacker, classifier, cls_optimizer, atk_optimizer,
                                       train_loader, args.iterations, epoch)
        WRITER.add_scalar('clean_loss', clean_loss, global_step=epoch - 1)
        WRITER.add_scalar('poison_loss', poison_loss, global_step=epoch - 1)
        scratchmodel = create_net(args).cuda()
        if epoch % args.test_interval == 0 or epoch == args.epochs:
            acc = test(args, attacker, scratchmodel,
                       train_loader, test_loader, epoch, trainepoch=args.test_epochs)
            if acc < best_acc:
                best_acc = acc
                torch.save(attacker.state_dict(), os.path.join(args.tensorboard_path,
                                                               "atk{}_best.pth".format(int(args.eps * 255))))
            if epoch % 10 == 0:
                torch.save(attacker.state_dict(), os.path.join(args.tensorboard_path,
                                                               "atk{}_e{}.pth".format(int(args.eps * 255), epoch)))
            torch.save(attacker.state_dict(), os.path.join(args.tensorboard_path,
                                                           "atk{}_latest.pth".format(int(args.eps * 255))))
        cls_schedule.step()
        atk_schedule.step()

