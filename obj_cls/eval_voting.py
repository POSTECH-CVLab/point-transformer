from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.data_util import ModelNet40 as ModelNet40
import numpy as np
from torch.utils.data import DataLoader
from util.util import IOStream, load_cfg_from_cfg_file, merge_cfg_from_list
import sklearn.metrics as metrics
import random


def get_parser():
    parser = argparse.ArgumentParser(description='3D Object Classification')
    parser.add_argument('--config', type=str, default='config/dgcnn_paconv_train.yaml', help='config file')
    parser.add_argument('opts', help='see config/dgcnn_paconv_train.yaml for all options', default=None, nargs=argparse.REMAINDER)
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

    # backup the running files:
    os.system('cp eval_voting.py checkpoints' + '/' + args.exp_name + '/' + 'eval_voting.py.backup')


class PointcloudScale(object):  # input random scaling
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            scales = torch.from_numpy(xyz).float().cuda()
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], scales)
        return pc


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, pt_norm=False), num_workers=args.workers,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    NUM_PEPEAT = 300
    NUM_VOTE = 10

    # Try to load models:
    if args.arch == 'dgcnn':
        from model.DGCNN_PAConv import PAConv
        model = PAConv(args).to(device)
    elif args.arch == 'pointnet':
        from model.PointNet_PAConv import PAConv
        model = PAConv(args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("checkpoints/%s/best_model.t7" % args.exp_name))
    model = model.eval()
    best_acc = 0

    pointscale = PointcloudScale(scale_low=0.8, scale_high=1.18)  # set the range of scaling

    for i in range(NUM_PEPEAT):
        test_true = []
        test_pred = []

        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            pred = 0
            for v in range(NUM_VOTE):
                new_data = data
                batch_size = data.size()[0]
                if v > 0:
                    new_data.data = pointscale(new_data.data)
                with torch.no_grad():
                    pred += F.softmax(model(new_data.permute(0, 2, 1)), dim=1)  # sum 10 preds
            pred /= NUM_VOTE   # avg the preds!
            label = label.view(-1)
            pred_choice = pred.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(pred_choice.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        if test_acc > best_acc:
            best_acc = test_acc
        outstr = 'Voting %d, test acc: %.6f,' % (i, test_acc * 100)
        io.cprint(outstr)

    final_outstr = 'Final voting test acc: %.6f,' % (best_acc * 100)
    io.cprint(final_outstr)


if __name__ == "__main__":
    args = get_parser()
    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/%s_voting.log' % (args.exp_name))
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

    test(args, io)
