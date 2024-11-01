"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import OPIXray_ROOT, OPIXrayAnnotationTransform, OPIXrayDetection, BaseTransform
from data import OPIXray_CLASSES as labelmap
from metrics.detection.mAP import *
import torch.utils.data as data
import warnings
warnings.filterwarnings("ignore")
from ssd import build_ssd
import sys

import os
import os.path as osp
import time
import argparse
import numpy as np
import pickle
import cv2
import shutil
import pylab as pl
import matplotlib.pyplot as plt
pl.mpl.rcParams['font.sans-serif'] = ['SimHei']
pl.mpl.rcParams['axes.unicode_minus'] = False
# -*- coding:utf-8 -*-
import importlib
importlib.reload(sys)
#os.environ['NLS_LANG'] = 'Simplified Chinese_CHINA.ZHS16GBK'

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "a1")


GPUID = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default=None, type=str,
                    help='Trained state_dict file path to open')
parser.add_argument(  # '--save_folder', default='/media/dsg3/husheng/eval/', type=str,
    '--save_folder',
    default="", type=str,
    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.25, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--OPIXray_root', default=OPIXray_ROOT,
                    help='Location of OPIXray root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--imagesetfile',
                    default=None, type=str,
                    help='imageset file path to open')

args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        # print("WARNING: It looks like you have a CUDA device, but aren't using \
        #         CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.OPIXray_root, 'test_annotation', '%s.xml')
imgpath = os.path.join(args.OPIXray_root, 'test_iamge', '%s.jpg')

# imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')

YEAR = '2007'

devkit_path = args.save_folder
dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    # //
    # //
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')
    # if(k==0):
    # img = x.int().cpu().squeeze().permute(1,2,0).detach().numpy()
    # cv2.imwrite('edge_s.jpg',img)
    #    x = self.edge_conv2d(x)
    # rgb_im = rgb_im.int().cpu().squeeze().permute(1,2,0).detach().numpy()
    # cv2.imwrite('rgb_im.jpg', rgb_im)
    # for i in range(6):
    #    im = Image.fromarray(edge_detect[i]*255).convert('L')
    #    im.save(str(i)+'edge.jpg')
    # x = self.edge_conv2d.edge_conv2d(x)
    # else:
    for i in range(num_images):
        im, gt, h, w, og_im = dataset.pull_item(i)
        #img = im.int().cpu().squeeze().permute(1, 2, 0).detach().numpy()
        #cv2.imwrite('/mnt/SSD/results/orgin'+str(i)+'.jpg', img)
        # im_saver = cv2.resize(im[(a2,a1,0),:,:].permute((a1,a2,0)).numpy(), (w,h))
        im = im.type(torch.FloatTensor)
        im_det = og_im.copy()
        im_gt = og_im.copy()

        # print(im_det)
        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        # //
        # //
        # print(detections)
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            # print(boxes)
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

            # print(all_boxes)
            #for item in cls_dets:
                # print(item)
                # print(item[5])
                #if item[4] > thresh:
                    # print(item)
                    #chinese = labelmap[j - 1] + str(round(item[4], 2))
                    # print(chinese+'det\n\n')
                    #if chinese[0] == 'knife':
                        #chinese = 'knife' + chinese[6:]
                    # cv2.rectangle(im_det, (item[0], item[1]), (item[2], item[3]), (0, 0, 255), 2)
                    # cv2.putText(im_det, chinese, (int(item[0]), int(item[1]) - 5), 0, 0.6, (0, 0, 255), 2)
        real = 0
        if gt[0][4] == 99:
            real = 0
        else:
            real = 1

        for item in gt:
            if real == 0:
                print('this pic dont have the obj:', dataset.ids[i])
                break



    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    # print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def reset_args():
    global args
    #args.trained_model = "/mnt/model/SSD/weights/Xray20190723/2020-2-16-16-38/knife_rgb_r_score_attention_adapt_sigmoid_rgb_red-rgb_position-gated_conv_hou/ssd300_Xray_knife_{:d}.pth".format(
    #    EPOCH)
    saver_root = 'save/'
    if not os.path.exists(saver_root):
        os.mkdir(saver_root)
    args.save_folder = saver_root + 'best_result_epoch/'

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    else:
        shutil.rmtree(args.save_folder)
        os.mkdir(args.save_folder)

    global devkit_path
    devkit_path = args.save_folder


if __name__ == '__main__':
    #EPOCHS = [45]
    #EPOCHS = [40,45,50, 55, 60, 65, 70, 75, 80,85,90,95,100,105,110,115,120,125,130,135,140,145]
    #EPOCHS = [130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,255]
    # EPOCHS = [90, 95, 100, 105, 110, 115, 120, 125]
    #EPOCHS = [255]
    #print(EPOCHS)
    #for EPOCH in EPOCHS:
    reset_args()

        # load net
    num_classes = len(labelmap) + 1  # +a1 for background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
        # print('Finished loading model!')
        # load data
    dataset = OPIXrayDetection(args.OPIXray_root, args.imagesetfile,
                                  #BaseTransform(300, dataset_mean),
                                  OPIXrayAnnotationTransform(),phase='test')
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
        # evaluation

    test_net(args.save_folder, net, args.cuda, dataset,
                 None, args.top_k, 300,
                 thresh=args.confidence_threshold)
