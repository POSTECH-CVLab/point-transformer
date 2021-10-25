from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from util.data_util import ModelNet40 as ModelNet40
import numpy as np
from util.util import cal_loss, load_cfg_from_cfg_file, merge_cfg_from_list, find_free_port, AverageMeter, intersectionAndUnionGPU
import time
import logging
import random
from tensorboardX import SummaryWriter


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    file_handler = logging.FileHandler(os.path.join('checkpoints', args.exp_name, 'main-' + str(int(time.time())) + '.log'))
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)

    return logger


def get_parser():
    parser = argparse.ArgumentParser(description='3D Object Classification')
    parser.add_argument('--config', type=str, default='config/dgcnn_paconv.yaml', help='config file')
    parser.add_argument('opts', help='see config/dgcnn_paconv.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    cfg['classes'] = cfg.get('classes', 40)
    cfg['sync_bn'] = cfg.get('sync_bn', True)
    cfg['dist_url'] = cfg.get('dist_url', 'tcp://127.0.0.1:6789')
    cfg['dist_backend'] = cfg.get('dist_backend', 'nccl')
    cfg['multiprocessing_distributed'] = cfg.get('multiprocessing_distributed', True)
    cfg['world_size'] = cfg.get('world_size', 1)
    cfg['rank'] = cfg.get('rank', 0)
    cfg['manual_seed'] = cfg.get('manual_seed', 0)
    cfg['workers'] = cfg.get('workers', 6)
    cfg['print_freq'] = cfg.get('print_freq', 10)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


# weight initialization:
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def train(gpu, ngpus_per_node):

    # ============= Model ===================
    if args.arch == 'dgcnn':
        from model.DGCNN_PAConv import PAConv
        model = PAConv(args)
    elif args.arch == 'pointnet':
        from model.PointNet_PAConv import PAConv
        model = PAConv(args)
    else:
        raise Exception("Not implemented")

    model.apply(weight_init)

    if main_process():
        logger.info(model)

    if args.sync_bn and args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model.cuda())

    # =========== Dataloader =================
    train_data = ModelNet40(partition='train', num_points=args.num_points, pt_norm=args.pt_norm)
    test_data = ModelNet40(partition='test', num_points=args.num_points, pt_norm=False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    # ============= Optimizer ===================
    if main_process():
        logger.info("Use SGD")
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr/100)
    
    criterion = cal_loss
    best_test_acc = 0
    start_epoch = 0

    # ============= Training from scratch=================
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_epoch(train_loader, model, opt, scheduler, epoch, criterion)

        test_acc = test_epoch(test_loader, model, epoch, criterion)

        if test_acc >= best_test_acc and main_process():
            best_test_acc = test_acc
            logger.info('Max Acc:%.6f' % best_test_acc)
            torch.save(model.state_dict(), 'checkpoints/%s/best_model.t7' % args.exp_name)  # save the best model


def train_epoch(train_loader, model, opt, scheduler, epoch, criterion):
    train_loss = 0.0
    count = 0.0

    batch_time = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    for ii, (data, label) in enumerate(train_loader):
        data_time.update(time.time() - end)

        data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True).squeeze(1)
        data = data.permute(0, 2, 1)
        batch_size = data.size(0)
        end2 = time.time()
        logits, loss = model(data, label, criterion)

        forward_time.update(time.time() - end2)

        preds = logits.max(dim=1)[1]

        if not args.multiprocessing_distributed:
            loss = torch.mean(loss)

        end3 = time.time()
        opt.zero_grad()
        loss.backward()   # the own loss of each process, backward by the optimizer belongs to this process
        opt.step()
        backward_time.update(time.time() - end3)

        # Loss
        if args.multiprocessing_distributed:
            loss = loss * batch_size
            _count = label.new_tensor([batch_size], dtype=torch.long).cuda(non_blocking=True)  # b_size on one process
            dist.all_reduce(loss), dist.all_reduce(_count)  # obtain the sum of all xxx at all processes
            n = _count.item()
            loss = loss / n   # avg loss across all processes

        # then calculate loss same as without dist
        count += batch_size
        train_loss += loss.item() * batch_size

        loss_meter.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + ii + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (ii + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Forward {for_time.val:.3f} ({for_time.avg:.3f}) '
                        'Backward {back_time.val:.3f} ({back_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '.format(epoch + 1, args.epochs, ii + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          for_time = forward_time,
                                                          back_time = backward_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter))

        intersection, union, target = intersectionAndUnionGPU(preds, label, args.classes)
        if args.multiprocessing_distributed:    # obtain the sum of all tensors at all processes: all_reduce
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

    scheduler.step()

    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)  # the first sum here is to sum the acc across all classes

    outstr = 'Train %d, loss: %.6f, train acc: %.6f, ' \
             'train avg acc: %.6f' % (epoch + 1,
                                      train_loss * 1.0 / count,
                                      allAcc, mAcc)

    if main_process():
        logger.info(outstr)
        # Write to tensorboard
        writer.add_scalar('loss_train', train_loss * 1.0 / count, epoch + 1)
        writer.add_scalar('mAcc_train', mAcc, epoch + 1)
        writer.add_scalar('allAcc_train', allAcc, epoch + 1)


def test_epoch(test_loader, model, epoch, criterion):
    test_loss = 0.0
    count = 0.0
    model.eval()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for data, label in test_loader:
        data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True).squeeze(1)
        data = data.permute(0, 2, 1)
        batch_size = data.size(0)
        logits = model(data)

        # Loss
        loss = criterion(logits, label)  # here use model's output directly
        if args.multiprocessing_distributed:
            loss = loss * batch_size
            _count = label.new_tensor([batch_size], dtype=torch.long).cuda(non_blocking=True)
            dist.all_reduce(loss), dist.all_reduce(_count)
            n = _count.item()
            loss = loss / n
        else:
            loss = torch.mean(loss)

        preds = logits.max(dim=1)[1]
        count += batch_size
        test_loss += loss.item() * batch_size

        intersection, union, target = intersectionAndUnionGPU(preds, label, args.classes)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    outstr = 'Test %d, loss: %.6f, test acc: %.6f, ' \
             'test avg acc: %.6f' % (epoch + 1,
                                     test_loss * 1.0 / count,
                                     allAcc,
                                     mAcc)

    if main_process():
        logger.info(outstr)
        # Write to tensorboard
        writer.add_scalar('loss_test', test_loss * 1.0 / count, epoch + 1)
        writer.add_scalar('mAcc_test', mAcc, epoch + 1)
        writer.add_scalar('allAcc_test', allAcc, epoch + 1)

    return allAcc


def test(gpu, ngpus_per_node):
    if main_process():
        logger.info('<<<<<<<<<<<<<<<<< Start Evaluation <<<<<<<<<<<<<<<<<')

    # ============= Model ===================
    if args.arch == 'dgcnn':
        from model.DGCNN_PAConv import PAConv
        model = PAConv(args)
    elif args.arch == 'pointnet':
        from model.PointNet_PAConv import PAConv
        model = PAConv(args)
    else:
        raise Exception("Not implemented")

    if main_process():
        logger.info(model)

    if args.sync_bn:
        assert args.distributed == True
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model.cuda())

    state_dict = torch.load("checkpoints/%s/best_model.t7" % args.exp_name, map_location=torch.device('cpu'))

    for k in state_dict.keys():
        if 'module' not in k:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k in state_dict:
                new_state_dict['module.' + k] = state_dict[k]
            state_dict = new_state_dict
        break
    
    model.load_state_dict(state_dict)

    # Dataloader
    test_data = ModelNet40(partition='test', num_points=args.num_points)
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    model.eval()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for data, label in test_loader:

        data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True).squeeze(1)
        data = data.permute(0, 2, 1)
        with torch.no_grad():
            logits = model(data)
        preds = logits.max(dim=1)[1]

        intersection, union, target = intersectionAndUnionGPU(preds, label, args.classes)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Test result: mAcc/allAcc {:.4f}/{:.4f}.'.format(mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: accuracy {:.4f}.'.format(i, accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    if main_process():
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if not os.path.exists('checkpoints/' + args.exp_name):
            os.makedirs('checkpoints/' + args.exp_name)

        if not args.eval:  # backup the running files
            os.system('cp main_ddp.py checkpoints' + '/' + args.exp_name + '/' + 'main_ddp.py.backup')
            os.system('cp util/PAConv_util.py checkpoints' + '/' + args.exp_name + '/' + 'PAConv_util.py.backup')
            os.system('cp util/data_util.py checkpoints' + '/' + args.exp_name + '/' + 'data_util.py.backup')
            if args.arch == 'dgcnn':
                os.system('cp model/DGCNN_PAConv.py checkpoints' + '/' + args.exp_name + '/' + 'DGCNN_PAConv.py.backup')
            elif args.arch == 'pointnet':
                os.system(
                    'cp model/PointNet_PAConv.py checkpoints' + '/' + args.exp_name + '/' + 'PointNet_PAConv.py.backup')

        global logger, writer
        writer = SummaryWriter('checkpoints/' + args.exp_name)
        logger = get_logger()
        logger.info(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert not args.eval, "The all_reduce function of PyTorch DDP will ignore/repeat inputs " \
                          "(leading to the wrong result), " \
                          "please use main.py to test (avoid DDP) for getting the right result."
    train(gpu, ngpus_per_node)


if __name__ == "__main__":
    args = get_parser()
    args.gpu = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.gpu)
    if len(args.gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.gpu, args.ngpus_per_node, args)

