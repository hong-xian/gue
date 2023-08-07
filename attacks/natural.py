import torch
import torch.nn as nn

from tqdm import tqdm
import sys
sys.path.append('..')

from utils import AverageMeter, accuracy_top1


@torch.no_grad()
def natural_attack(args, model, loader, writer=None, epoch=0, loop_type='test'):
    model.eval()
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()

    iterator = tqdm(enumerate(loader), total=len(loader), ncols=110)

    count = []

    for i, (inp, target) in iterator:
        inp = inp.cuda()
        target = target.cuda()

        logits = model(inp)

        count.append(logits.argmax(1))
        loss = nn.CrossEntropyLoss()(logits, target)
        acc = accuracy_top1(logits, target)

        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))

        desc = ('[{}] | Loss {:.4f} | Accuracy {:.4f} ||'
                .format(loop_type, loss_logger.avg, acc_logger.avg))
        iterator.set_description(desc)
    #
    # count = torch.cat(count)
    # count = torch.bincount(count)
    # print(count.data)

    if writer is not None:
        # descs = ['loss', 'accuracy']
        # vals = [loss_logger, acc_logger]
        # for k, v in zip(descs, vals):
        #     writer.add_scalar('cln_{}_{}'.format(loop_type, k), v.avg, epoch)
        writer.add_scalar('nat_{}_acc'.format(loop_type), acc_logger.avg, epoch)

    return loss_logger.avg, acc_logger.avg
