from __future__ import print_function
import os
import argparse
import torch
from util.data_util import PartNormalDataset
from model.DGCNN_PAConv_vote import PAConv
import numpy as np
from torch.utils.data import DataLoader
from util.util import to_categorical, compute_overall_iou, load_cfg_from_cfg_file, merge_cfg_from_list, IOStream
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import torch.nn.functional as F

classes_str =['aero','bag','cap','car','chair','ear','guitar','knife','lamp','lapt','moto','mug','Pistol','rock','stake','table']


class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz).float().cuda())
        return pc


def get_parser():
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--config', type=str, default='dgcnn_paconv_test.yaml', help='config file')
    parser.add_argument('opts', help='see config/dgcnn_paconv_test.yaml for all options', default=None, nargs=argparse.REMAINDER)
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
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)

    # backup the running files:
    os.system('cp eval_voting.py checkpoints' + '/' + args.exp_name + '/' + 'eval_voting.py.backup')


def test(args, io):
    # Try to load models
    num_part = 50
    device = torch.device("cuda" if args.cuda else "cpu")

    model = PAConv(args, num_part).to(device)
    io.cprint(str(model))

    from collections import OrderedDict
    state_dict = torch.load("checkpoints/%s/best_insiou_model.pth" % args.exp_name,
                            map_location=torch.device('cpu'))['model']

    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)

    # Dataloader
    test_data = PartNormalDataset(npoints=2048, split='test', normalize=False)
    print("The number of test data is:%d", len(test_data))

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                             drop_last=False)

    NUM_PEPEAT = 100
    NUM_VOTE = 10
    global_Class_mIoU, global_Inst_mIoU = 0, 0
    global_total_per_cat_iou = np.zeros((16)).astype(np.float32)
    num_part = 50
    num_classes = 16
    pointscale = PointcloudScale(scale_low=0.87, scale_high=1.15)

    model.eval()

    for i in range(NUM_PEPEAT):

        metrics = defaultdict(lambda: list())
        shape_ious = []
        total_per_cat_iou = np.zeros((16)).astype(np.float32)
        total_per_cat_seen = np.zeros((16)).astype(np.int32)

        for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(test_loader), total=len(test_loader),
                                                                smoothing=0.9):
            batch_size, num_point, _ = points.size()
            points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(
                target.long()), Variable(norm_plt.float())
            # points = points.transpose(2, 1)
            norm_plt = norm_plt.transpose(2, 1)
            points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze().cuda(
                non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)

            seg_pred = 0
            new_points = Variable(torch.zeros(points.size()[0], points.size()[1], points.size()[2]).cuda(),
                                  volatile=True)

            for v in range(NUM_VOTE):
                if v > 0:
                    new_points.data = pointscale(points.data)
                with torch.no_grad():
                    seg_pred += F.softmax(
                        model(points.contiguous().transpose(2, 1), new_points.contiguous().transpose(2, 1),
                              norm_plt, to_categorical(label, num_classes)), dim=2)  # xyz,x: only scale feature input
            seg_pred /= NUM_VOTE

            # instance iou without considering the class average at each batch_size:
            batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
            shape_ious += batch_shapeious  # iou +=, equals to .append

            # per category iou at each batch_size:
            for shape_idx in range(seg_pred.size(0)):  # sample_idx
                cur_gt_label = label[shape_idx]  # label[sample_idx]
                total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]
                total_per_cat_seen[cur_gt_label] += 1

            # accuracy:
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            metrics['accuracy'].append(correct.item() / (batch_size * num_point))

        metrics['shape_avg_iou'] = np.mean(shape_ious)
        for cat_idx in range(16):
            if total_per_cat_seen[cat_idx] > 0:
                total_per_cat_iou[cat_idx] = total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx]

        print('\n------ Repeat %3d ------' % (i + 1))

        # First we need to calculate the iou of each class and the avg class iou:
        class_iou = 0
        for cat_idx in range(16):
            class_iou += total_per_cat_iou[cat_idx]
            io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))  # print the iou of each class
        avg_class_iou = class_iou / 16
        outstr = 'Test :: test class mIOU: %f, test instance mIOU: %f' % (avg_class_iou, metrics['shape_avg_iou'])
        io.cprint(outstr)

        if avg_class_iou > global_Class_mIoU:
            global_Class_mIoU = avg_class_iou
            global_total_per_cat_iou = total_per_cat_iou

        if metrics['shape_avg_iou'] > global_Inst_mIoU:
            global_Inst_mIoU = metrics['shape_avg_iou']

    # final avg print:
    final_out_str = 'Best voting result :: test class mIOU: %f, test instance mIOU: %f' % (global_Class_mIoU, global_Inst_mIoU)
    io.cprint(final_out_str)

    # final per cat print:
    for cat_idx in range(16):
        io.cprint(classes_str[cat_idx] + ' iou: ' + str(global_total_per_cat_iou[cat_idx]))  # print iou of each class


if __name__ == "__main__":
    args = get_parser()
    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/%s_voting.log' % (args.exp_name))
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.manual_seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.manual_seed)
    else:
        io.cprint('Using CPU')

    test(args, io)

