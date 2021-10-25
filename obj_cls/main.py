from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from util.data_util import ModelNet40 as ModelNet40
import numpy as np
from torch.utils.data import DataLoader
from util.util import cal_loss, IOStream, load_cfg_from_cfg_file, merge_cfg_from_list
import sklearn.metrics as metrics
from tensorboardX import SummaryWriter
import random


def get_parser():
    parser = argparse.ArgumentParser(description='3D Object Classification')
    parser.add_argument('--config', type=str, default='config/dgcnn_paconv.yaml', help='config file')
    parser.add_argument('opts', help='see config/dgcnn_paconv.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    cfg['manual_seed'] = cfg.get('manual_seed', 0)
    cfg['workers'] = cfg.get('workers', 6)
    return cfg


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)

    if not args.eval:  # backup the running files
        os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
        os.system('cp util/PAConv_util.py checkpoints' + '/' + args.exp_name + '/' + 'PAConv_util.py.backup')
        os.system('cp util/data_util.py checkpoints' + '/' + args.exp_name + '/' + 'data_util.py.backup')
        if args.arch == 'dgcnn':
            os.system('cp model/DGCNN_PAConv.py checkpoints' + '/' + args.exp_name + '/' + 'DGCNN_PAConv.py.backup')
        elif args.arch == 'pointnet':
            os.system('cp model/PointNet_PAConv.py checkpoints' + '/' + args.exp_name + '/' + 'PointNet_PAConv.py.backup')

    global writer
    writer = SummaryWriter('checkpoints/' + args.exp_name)


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


def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points, pt_norm=args.pt_norm),
                              num_workers=args.workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, pt_norm=False),
                             num_workers=args.workers, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.arch == 'dgcnn':
        from model.DGCNN_PAConv import PAConv
        model = PAConv(args).to(device)
    elif args.arch == 'pointnet':
        from model.PointNet_PAConv import PAConv
        model = PAConv(args).to(device)
    else:
        raise Exception("Not implemented")

    io.cprint(str(model))

    model.apply(weight_init)
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    print("Use SGD")
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr/100)

    criterion = cal_loss

    best_test_acc = 0

    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, ' % (epoch, train_loss * 1.0 / count, train_acc)
        io.cprint(outstr)

        writer.add_scalar('loss_train', train_loss * 1.0 / count, epoch + 1)
        writer.add_scalar('Acc_train', train_acc, epoch + 1)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f,' % (epoch, test_loss * 1.0 / count, test_acc)
        io.cprint(outstr)

        writer.add_scalar('loss_test', test_loss * 1.0 / count, epoch + 1)
        writer.add_scalar('Acc_test', test_acc, epoch + 1)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            io.cprint('Max Acc:%.6f' % best_test_acc)
            torch.save(model.state_dict(), 'checkpoints/%s/best_model.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, pt_norm=False),
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models:
    if args.arch == 'dgcnn':
        from model.DGCNN_PAConv import PAConv
        model = PAConv(args).to(device)
    elif args.arch == 'pointnet':
        from model.PointNet_PAConv import PAConv
        model = PAConv(args).to(device)
    else:
        raise Exception("Not implemented")

    io.cprint(str(model))

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("checkpoints/%s/best_model.t7" % args.exp_name))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        with torch.no_grad():
            logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    args = get_parser()
    _init_()

    if not args.eval:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_train.log' % (args.exp_name))
    else:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_test.log' % (args.exp_name))
    io.cprint(str(args))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
