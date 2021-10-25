import os
import numpy as np
import pickle5 as pickle
import logging

from util.util import AverageMeter, intersectionAndUnion, check_makedirs


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def get_color(i):
    ''' Parse a 24-bit integer as a RGB color. I.e. Convert to base 256
    Args:
        index: An int. The first 24 bits will be interpreted as a color.
            Negative values will not work properly.
    Returns:
        color: A color s.t. get_index( get_color( i ) ) = i
    '''
    b = (i) % 256  # least significant byte
    g = (i >> 8) % 256
    r = (i >> 16) % 256 # most significant byte
    return r, g, b


def main():
    global logger
    logger = get_logger()

    classes = 13
    color_map = np.zeros((classes, 3))
    names = [line.rstrip('\n') for line in open('data/s3dis/s3dis_names.txt')]
    for i in range(classes):
        color_map[i, :] = get_color(i)
    data_root = 'dataset/s3dis/trainval_fullarea'
    data_list = sorted(os.listdir(data_root))
    data_list = [item[:-4] for item in data_list if 'Area_' in item]
    intersection_meter, union_meter, target_meter = AverageMeter(), AverageMeter(), AverageMeter()

    logger.info('<<<<<<<<<<<<<<<<< Start Evaluation <<<<<<<<<<<<<<<<<')
    test_area = [1, 2, 3, 4, 5, 6]
    for i in range(len(test_area)):
        # result_path = os.path.join('exp/s3dis', exp_list[test_area[i]-1], 'result')
        result_path = '/exp/s3dis/6-fold'  # where to save all result files
        # pred_save_folder = os.path.join(result_path, 'best_visual/pred')
        # label_save_folder = os.path.join(result_path, 'best_visual/label')
        # image_save_folder = os.path.join(result_path, 'best_visual/image')
        # check_makedirs(pred_save_folder); check_makedirs(label_save_folder); check_makedirs(image_save_folder)
        with open(os.path.join(result_path, 'pred_{}'.format(test_area[i]) + '.pickle'), 'rb') as handle:
            pred = pickle.load(handle)['pred']
        with open(os.path.join(result_path, 'gt_{}'.format(test_area[i]) + '.pickle'), 'rb') as handle:
            label = pickle.load(handle)['gt']
        data_split = [item for item in data_list if 'Area_{}'.format(test_area[i]) in item]
        assert len(pred) == len(label) == len(data_split)
        for j in range(len(data_split)):
            print('processing [{}/{}]-[{}/{}]'.format(i+1, len(test_area), j+1, len(data_split)))
            # data_name = data_split[j]
            # data = np.load(os.path.join(data_root, data_name + '.npy'))
            # coord, feat = data[:, :3], data[:, 3:6]
            pred_j, label_j = pred[j].astype(np.uint8), label[j].astype(np.uint8)
            # pred_j_color, label_j_color = color_map[pred_j, :], color_map[label_j, :]
            # vis_util.write_ply_color(coord, pred_j, os.path.join(pred_save_folder, data_name +'.obj'))
            # vis_util.write_ply_color(coord, label_j, os.path.join(label_save_folder, data_name + '.obj'))
            # vis_util.write_ply_rgb(coord, feat, os.path.join(image_save_folder, data_name + '.obj'))
            intersection, union, target = intersectionAndUnion(pred_j, label_j, classes, ignore_index=255)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

    for i in range(classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
