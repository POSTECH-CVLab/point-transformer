import os
import time
import random
import numpy as np
import subprocess

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import dataset, transform
from util.s3dis import S3DIS
from util.util import AverageMeter, intersectionAndUnionGPU, get_logger, get_parser
from model.pointnet2.paconv import PAConv


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def init():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    if args.train_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    if args.train_gpu is not None and len(args.train_gpu) == 1:
        args.sync_bn = False
    logger.info(args)


def get_git_commit_id():
    if not os.path.exists('.git'):
        return '0000000'
    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_id = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_id


def main():
    init()
    if args.arch == 'pointnet_seg':
        from model.pointnet.pointnet import PointNetSeg as Model
    elif args.arch == 'pointnet2_seg':
        from model.pointnet2.pointnet2_seg import PointNet2SSGSeg as Model
    elif args.arch == 'pointnet2_paconv_seg':
        from model.pointnet2.pointnet2_paconv_seg import PointNet2SSGSeg as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(c=args.fea_dim, k=args.classes, use_xyz=args.use_xyz, args=args)

    best_mIoU = 0.0

    if args.sync_bn:
        from util.util import convert_to_syncbn
        convert_to_syncbn(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.get('lr_multidecay', False):
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs * 0.6), int(args.epochs * 0.8)], gamma=args.multiplier)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_epoch, gamma=args.multiplier)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    model = torch.nn.DataParallel(model.cuda())
    if args.sync_bn:
        from lib.sync_bn import patch_replication_callback
        patch_replication_callback(model)
    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            try:
                best_mIoU = checkpoint['val_mIoU']
            except Exception:
                pass
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.get('no_transformation', True):
        train_transform = None
    else:
        train_transform = transform.Compose([transform.RandomRotate(along_z=args.get('rotate_along_z', True)),
                                             transform.RandomScale(scale_low=args.get('scale_low', 0.8), 
                                                                   scale_high=args.get('scale_high', 1.2)),
                                             transform.RandomJitter(sigma=args.get('jitter_sigma', 0.01),
                                                                    clip=args.get('jitter_clip', 0.05)),
                                             transform.RandomDropColor(color_augment=args.get('color_augment', 0.0))])
    logger.info(train_transform)
    if args.data_name == 's3dis':
        train_data = S3DIS(split='train', data_root=args.train_full_folder, num_point=args.num_point, 
                           test_area=args.test_area, block_size=args.block_size, sample_rate=args.sample_rate, transform=train_transform,
                           fea_dim=args.get('fea_dim', 6), shuffle_idx=args.get('shuffle_idx', False))
    else:
        raise ValueError('{} dataset not supported.'.format(args.data_name))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_workers, pin_memory=True, drop_last=True)

    val_loader = None
    if args.evaluate:
        val_transform = transform.Compose([transform.ToTensor()])
        if args.data_name == 's3dis':
            val_data = dataset.PointData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform, 
                                         norm_as_feat=args.get('norm_as_feat', True), fea_dim=args.get('fea_dim', 6))
        else:
            raise ValueError('{} dataset not supported.'.format(args.data_name))      

        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.train_batch_size_val, shuffle=False, num_workers=args.train_workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch, args.get('correlation_loss', False))
        epoch_log = epoch + 1
        writer.add_scalar('loss_train', loss_train, epoch_log)
        writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
        writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
        writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        if epoch_log % args.save_freq == 0:
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 
                        'commit_id': get_git_commit_id()}, filename)
            if epoch_log / args.save_freq > 2:
                try:
                    deletename = args.save_path + '/train_epoch_' + str(epoch_log - args.save_freq * 2) + '.pth'
                    os.remove(deletename)
                except Exception:
                    logger.info('{} Not found.'.format(deletename))

        if args.evaluate and epoch_log % args.get('eval_freq', 1) == 0:
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
            writer.add_scalar('loss_val', loss_val, epoch_log)
            writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
            writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
            writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
            if mIoU_val > best_mIoU:
                best_mIoU = mIoU_val
                filename = args.save_path + '/best_train.pth'
                logger.info('Best Model Saving checkpoint to: ' + filename)
                torch.save(
                    {'epoch': epoch_log, 'state_dict': model.state_dict(), 
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                     'val_mIoU': best_mIoU, 'commit_id': get_git_commit_id()}, filename)
        scheduler.step()


def train(train_loader, model, criterion, optimizer, epoch, correlation_loss):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    main_loss_meter = AverageMeter()
    corr_loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        main_loss = criterion(output, target)

        corr_loss = 0.0
        corr_loss_scale = args.get('correlation_loss_scale', 10.0)
        if correlation_loss:
            for m in model.module.SA_modules.named_modules():
                if isinstance(m[-1], PAConv):
                    kernel_matrice, output_dim, m_dim = m[-1].weightbank, m[-1].output_dim, m[-1].m
                    new_kernel_matrice = kernel_matrice.view(-1, m_dim, output_dim).permute(1, 0, 2).reshape(m_dim, -1)
                    cost_matrice = torch.matmul(new_kernel_matrice, new_kernel_matrice.T) / torch.matmul(
                        torch.sqrt(torch.sum(new_kernel_matrice ** 2, dim=-1, keepdim=True)),
                        torch.sqrt(torch.sum(new_kernel_matrice.T ** 2, dim=0, keepdim=True)))
                    corr_loss += torch.sum(torch.triu(cost_matrice, diagonal=1) ** 2)
        loss = main_loss + corr_loss_scale * corr_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        main_loss_meter.update(main_loss.item(), input.size(0))
        corr_loss_meter.update(corr_loss.item() * corr_loss_scale if correlation_loss else corr_loss, input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Main Loss {main_loss_meter.val:.4f} '
                        'Corr Loss {corr_loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          main_loss_meter=main_loss_meter,
                                                          corr_loss_meter=corr_loss_meter,
                                                          accuracy=accuracy))

        writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
        writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
        writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
        writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        output = model(input)
        loss = criterion(output, target)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()
