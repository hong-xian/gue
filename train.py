import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy_top1
from attacks.natural import natural_attack
from attacks.adv import adv_attack, batch_adv_attack


def standard_loss(args, model, x, y):
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    return loss, logits


def adv_loss(args, model, x, y):
    model.eval()
    x_adv = batch_adv_attack(args, model, x, y)
    model.train()

    logits_adv = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits_adv, y)
    return loss, logits_adv


LOSS_FUNC = {
    '': standard_loss,
    'ST': standard_loss,
    'AT': adv_loss,
}


def train(args, model, optimizer, loader, writer, epoch):
    model.train()
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()

    iterator = tqdm(enumerate(loader), total=len(loader), ncols=95)
    for i, (inp, target) in iterator:
        inp = inp.cuda()
        target = target.cuda()

        loss, logits = LOSS_FUNC[args.train_loss](args, model, inp, target)
        acc = accuracy_top1(logits, target)

        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        desc = 'Train Epoch: {} | Loss {:.4f} | Accuracy {:.4f} ||'.format(epoch, loss_logger.avg, acc_logger.avg)
        iterator.set_description(desc)

    if writer is not None:
        descs = ['loss', 'accuracy']
        vals = [loss_logger, acc_logger]
        for d, v in zip(descs, vals):
            writer.add_scalar('train_{}'.format(d), v.avg, epoch)

    return loss_logger.avg, acc_logger.avg


def train_model(args, model, optimizer, train_loader, test_loader, writer, schedule, resume=0):
    for epoch in range(resume+1, args.epochs+1):
        train(args, model, optimizer, train_loader, writer, epoch)

        last_epoch = (epoch == (args.epochs - 1))
        should_log = (epoch % args.log_gap == 0)

        if should_log or last_epoch:
            # nat_clean_train_loss, nat_clean_train_acc = natural_attack(
            #     args, model, train_loader, writer, epoch, 'clean_train')
            nat_clean_test_loss, nat_clean_test_acc = natural_attack(
                args, model, test_loader, writer, epoch, 'clean_test')

            robust_target = (args.train_loss in ['AT', 'TRADES', 'MART'])
            # if robust_target:
                # adv_clean_train_loss, adv_clean_train_acc, _ = adv_attack(
                #     args, model, train_loader, writer, epoch, 'clean_train')
                # adv_clean_test_loss, adv_clean_test_acc, _ = adv_attack(
                #     args, model, test_loader, writer, epoch, 'clean_test')
            # else:
            #     adv_clean_train_loss, adv_clean_train_acc, adv_clean_test_loss, adv_clean_test_acc = -1, -1, -1, -1

            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'train_acc': -1,
                'train_loss': -1,
                # 'nat_clean_train_acc': nat_clean_train_acc,
                'nat_clean_test_acc': nat_clean_test_acc,
                # 'adv_clean_train_acc': adv_clean_train_acc,
                # 'adv_clean_test_acc': adv_clean_test_acc,
            }
            torch.save(checkpoint, args.model_path)
        schedule.step()
    return model


def poison_train_model(args, model, optimizer, poison_train_loader,
                       clean_test_loader, schedule, writer):
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train(args, model, optimizer, poison_train_loader, writer, epoch)
        last_epoch = (epoch == (args.epochs - 1))
        should_log = (epoch % args.log_gap == 0)

        if should_log or last_epoch:
            # nat_clean_train_loss, nat_clean_train_acc = natural_attack(
            #     args, model, clean_train_loader, writer, epoch, 'clean_train')
            nat_clean_test_loss, nat_clean_test_acc = natural_attack(
                args, model, clean_test_loader, writer, epoch, 'clean_test')
            # nat_tar_test_loss, nat_tar_test_acc = tar_attack(
            #     args, model, clean_test_loader, writer, epoch, 'clean_tar')
            # nat_poison_train_loss, nat_poison_train_acc = natural_attack(
            #     args, model, poison_train_loader, writer, epoch, 'poison_train')

            # robust_target = (args.train_loss in ['AT', 'TRADES'])
            # if robust_target:
                # adv_clean_train_loss, adv_clean_train_acc, _ = adv_attack(
                #     args, model, clean_train_loader, writer, epoch, 'clean_train')
                # adv_clean_test_loss, adv_clean_test_acc, _ = adv_attack(
                #     args, model, clean_test_loader, writer, epoch, 'clean_test')
                # adv_poison_train_loss, adv_poison_train_acc, _ = adv_attack(
                #     args, model, poison_train_loader, writer, epoch, 'poison_train')

            # else:
            #     adv_clean_test_acc = -1
            #     adv_poison_train_acc = -1

            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'nat_clean_test_acc': nat_clean_test_acc,
                # 'nat_tar_test_acc': nat_tar_test_acc,
                # 'nat_poison_train_acc': nat_poison_train_acc,
                # 'adv_clean_train_acc': adv_clean_train_acc,
                # 'adv_clean_test_acc': adv_clean_test_acc,
                # 'adv_poison_train_acc': adv_poison_train_acc,
            }
            torch.save(checkpoint, args.model_path)
        schedule.step()
    return model


def eval_model(args, model, loader):
    model.eval()
    args.eps = args.eps

    keys, values = [], []
    keys.append('Model')
    values.append(args.tensorboard_path)

    # Natural
    acc, name = natural_attack(args, model, loader)
    keys.append(name)
    values.append(acc)


    # Save results
    import csv
    csv_fn = '{}.csv'.format(args.tensorboard_path)
    with open(csv_fn, 'w') as f:
        write = csv.writer(f)
        write.writerow(keys)
        write.writerow(values)

    print('=> csv file is saved at [{}]'.format(csv_fn))
