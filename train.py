import os
import shutil
import json
import time
from apex import amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from toolbox import get_dataset # loss
from toolbox.optim.Ranger import Ranger
from toolbox import get_logger
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import save_ckpt
from toolbox.datasets.irseg import IRSeg
from toolbox.datasets.pst900 import PSTSeg
from toolbox.losses import lovasz_softmax

class eeemodelLoss(nn.Module):

    def __init__(self, class_weight=None, ignore_index=-100, reduction='mean'):
        super(eeemodelLoss, self).__init__()

        self.class_weight_semantic = torch.from_numpy(np.array(
            [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])).float()
        self.class_weight_binary = torch.from_numpy(np.array([1.5121, 10.2388])).float()
        self.class_weight_boundary = torch.from_numpy(np.array([1.4459, 23.7228])).float()

        #self.class_weight = class_weight
        #self.cross_entropy = nn.CrossEntropyLoss()

        self.sal_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)
        self.binary_loss = nn.CrossEntropyLoss(weight=self.class_weight_binary)
        self.boundary_loss = nn.CrossEntropyLoss(weight=self.class_weight_boundary)

    def forward(self, inputs, targets):
        semantic_gt, binary_gt, boundary_gt = targets
        out = inputs

        loss1 = self.sal_loss(out, semantic_gt)
        loss2 = lovasz_softmax(F.softmax(out, dim=1), semantic_gt, ignore=255)

        loss = loss1 + loss2
        return loss


def run(args):
    #torch.cuda.set_device(args.cuda)
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}-{cfg["dataset"]}-{cfg["model_name"]}-'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda')
    model = get_model(cfg)                                                                                                    
    model.to(device)

    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: % .4fM' % (total / 1e6))

    trainset, _, testset = get_dataset(cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)
    test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                             pin_memory=True)

    params_list = model.parameters()
    optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / (cfg['epochs']) ** 0.9))

    train_criterion = eeemodelLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()
    running_metrics_test = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    best_test = 0

    amp.register_float_function(torch, 'sigmoid')
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    #if torch.cuda.device_count() > 1:
        #print("Let's use", torch.cuda.device_count(), "GPUs!")
        #model = nn.DataParallel(model)


    for ep in range(cfg['epochs']):

        # training
        model.train()
        train_loss_meter.reset()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()

            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            label = sample['label'].to(device)
            bound = sample['bound'].to(device)
            binary_label = sample['binary_label'].to(device)
            targets = [label, binary_label, bound]
            predict = model(image, depth)

            #print(predict.shape, targets[0].shape)  #torch.Size([2, 9, 480, 640]) torch.Size([2, 480, 640])
            loss = train_criterion(predict, targets)
            ####################################################

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())

        scheduler.step(ep)

        # test
        with torch.no_grad():
            model.eval()
            running_metrics_test.reset()
            test_loss_meter.reset()
            for i, sample in enumerate(test_loader):

                image = sample['image'].to(device)
                # Here, depth is TIR.
                depth = sample['depth'].to(device)
                label = sample['label'].to(device)
                predict = model(image, depth)

                #label = label.float()
                #print(predict.shape, label.shape)   #torch.Size([9, 480, 640]) torch.Size([2, 480, 640])
                loss = criterion(predict, label)
                test_loss_meter.update(loss.item())

                predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
                label = label.cpu().numpy()
                running_metrics_test.update(label, predict)

        train_loss = train_loss_meter.avg
        test_loss = test_loss_meter.avg

        test_macc = running_metrics_test.get_scores()[0]["class_acc: "]
        test_miou = running_metrics_test.get_scores()[0]["mIou: "]
        test_avg = (test_macc + test_miou) / 2

        logger.info(
            f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] loss={train_loss:.3f}/{test_loss:.3f}, mPA={test_macc:.3f}, miou={test_miou:.3f}, avg={test_avg:.3f}')
        if test_avg > best_test and ep > 150:
            best_test = test_avg
            save_ckpt(logdir, model,ep+1)
            logger.info(
            	f'Save Iter = [{ep + 1:3d}],  mPA={test_macc:.3f}, miou={test_miou:.3f}, avg={test_avg:.3f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="configs/HFFNet.json", help="Configuration file to use")
    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--inputs", type=str.lower, default='rgb', choices=['rgb', 'rgbt'])
    parser.add_argument("--resume", type=str, default='',
                        help="use this file to load last checkpoint for continuing training")

    args = parser.parse_args()

    print("Starting Training!")
    run(args)
