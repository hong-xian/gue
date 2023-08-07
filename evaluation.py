import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import argparse
from PIL import Image
import os
import numpy as np
from pprint import pprint
from train import poison_train_model, train_model, eval_model
from utils import make_and_restore_model, setup_seed
from models.unet import UNet


def poison_data(args, atknet, train_loader):
    atknet.eval()
    poisoned_input = []
    clean_target = []
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            noise = atknet(data) * args.eps
            poisoned_data = torch.clamp(data + noise, 0, 1)
            poisoned_input.append(poisoned_data.detach().cpu())
            clean_target.append(target.detach().cpu())

    poisoned_input = torch.cat(poisoned_input, dim=0)
    clean_target = torch.cat(clean_target, dim=0)
    return poisoned_input, clean_target


class PoisonDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None):
        self.transform = transform
        self.data, self.targets = data, target
        self.data = self.data.permute(0, 2, 3, 1)  # convert to HWC
        self.data = (self.data * 255).type(torch.uint8)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy())
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


def get_poisoned_data(args, atknet, train_loader):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    poisoned_input, clean_target = poison_data(args, atknet, train_loader)
    poisoned_train_set = PoisonDataset(poisoned_input, clean_target, transform_train)
    poisoned_train_loader = DataLoader(poisoned_train_set, batch_size=args.batch_size, shuffle=True)
    return poisoned_train_loader


def make_data_clean(args):
    if args.clean_train:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()
    if args.dataset == "cifar10":
        train_set = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_test)
    elif args.dataset == "cifar100":
        train_set = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_test)
    elif args.dataset == "SVHN":
        train_set = datasets.SVHN(args.data_path, split="train", transform=transform_train)
        test_set = datasets.SVHN(args.data_path, split="test", transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


def main(args, atknet):
    clean_train_loader, clean_test_loader = make_data_clean(args)
    poison_train_loader = get_poisoned_data(args, atknet, clean_train_loader)
    model = make_and_restore_model(args.arch, args.dataset)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    writer = SummaryWriter(args.tensorboard_path)
    schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_step)
    poison_train_model(args, model, optimizer, poison_train_loader, clean_test_loader, schedule, writer)
    eval_model(args, model, clean_test_loader)


def clean_train(args):
    clean_train_loader, clean_test_loader = make_data_clean(args)
    model = make_and_restore_model(args.arch, args.dataset)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    writer = SummaryWriter(args.tensorboard_path)
    schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_step)
    train_model(args, model, optimizer, clean_train_loader, clean_test_loader, writer, schedule=schedule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training classifiers on  poisoned dataset')
    parser.add_argument('--clean_train', action='store_true', default=False, help="training on clean or poison")
    parser.add_argument('--gpu_id', default="2", type=str)
    parser.add_argument('--out_dir', default='./results', type=str)
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', "SVHN"])
    parser.add_argument('--train_loss', default='ST', type=str, choices=['ST', 'AT', 'mixup', "cutmix"])
    parser.add_argument('--eps', default=8, type=float)
    parser.add_argument('--eps_at', default=8, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--arch', default='ResNet18', type=str,
                        choices=['VGG16', 'ResNet18', 'ResNet50', 'DenseNet121', 'WRN28-10', "VGG11"])
    parser.add_argument('--constraint', default='Linf', type=str, choices=['Linf', 'L2'])
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--exp_name', default="gue", type=str, help='the tensorboard name')
    parser.add_argument('--path', default="./atkmodel", help='load atkmodel from checkpoint')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    # Training options
    args.eps = args.eps / 255
    args.batch_size = 256
    args.weight_decay = 5e-4
    args.log_gap = 1
    args.lr_milestones = [75, 90]
    args.lr_step = 0.1

    # Attack options
    args.eps_at = args.eps_at / 255
    args.step_size = args.eps_at / 4
    args.num_steps = 10
    args.random_restarts = 1

    # Miscellaneous
    args.out_dir = os.path.join(args.out_dir, args.dataset)
    args.data_path = os.path.join(args.data_path, args.dataset)
    args.exp_name = os.path.join(args.arch, args.exp_name)
    args.tensorboard_path = os.path.join(args.out_dir, args.exp_name, 'tensorboard')
    args.model_path = os.path.join(args.out_dir, args.exp_name, 'checkpoint.pth')

    pprint(vars(args))
    setup_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.clean_train:
        clean_train(args)
    else:
        atknet = UNet(3).cuda()
        atknet.load_state_dict(torch.load(args.path))
        atknet.eval()
        main(args, atknet)