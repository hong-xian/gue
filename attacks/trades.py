import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import sys

sys.path.append('..')

from utils import AverageMeter, accuracy_top1, accuracy
from attacks.step import LinfStep, L2Step

STEPS = {
    'Linf': LinfStep,
    'L2': L2Step,
}


def batch_trades_attack(args, model, x, target):
    orig_x = x.clone().detach()
    logits_cln = model(orig_x).detach().requires_grad_(False)
    step = STEPS[args.constraint](orig_x, args.eps_at, args.step_size)
    kl = nn.KLDivLoss(reduction='batchmean')

    def get_adv_examples(x):
        for _ in range(args.num_steps):
            x = x.clone().detach().requires_grad_(True)
            logits_adv = model(x)
            loss = -1 * kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_cln, dim=1))
            grad = torch.autograd.grad(loss, [x])[0]
            with torch.no_grad():
                x = step.step(x, grad)
                x = step.project(x)
                x = torch.clamp(x, 0, 1)
        return x.clone().detach()

    to_ret = None

    if args.random_restarts == 0:
        adv = get_adv_examples(x)
        to_ret = adv.detach()
    elif args.random_restarts == 1:
        x = x.detach() + 0.01 * torch.randn_like(x).detach()
        x = torch.clamp(x, 0, 1)
        adv = get_adv_examples(x)
        to_ret = adv.detach()

    return to_ret.detach().requires_grad_(False)


def batch_trades_delta(args, model, x, target):
    orig_x = x.clone().detach()
    logits_cln = model(orig_x).detach().requires_grad_(False)
    step = STEPS[args.constraint](orig_x, args.eps_at, args.step_size)
    kl = nn.KLDivLoss(reduction='batchmean')

    def get_adv_delta(x):
        for _ in range(args.num_steps):
            x = x.clone().detach().requires_grad_(True)
            logits_adv = model(x)
            loss = -1 * kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_cln, dim=1))
            grad = torch.autograd.grad(loss, [x])[0]
            with torch.no_grad():
                x = step.step(x, grad)
                x = step.project(x)
                x = torch.clamp(x, 0, 1)
                delta = x - orig_x
        return delta.clone().detach()

    to_ret = None

    if args.random_restarts == 0:
        adv = get_adv_delta(x)
        to_ret = adv.detach()
    elif args.random_restarts == 1:
        x = x.detach() + 0.01 * torch.randn_like(x).detach()
        x = torch.clamp(x, 0, 1)
        adv = get_adv_delta(x)
        to_ret = adv.detach()

    return to_ret.detach().requires_grad_(False)