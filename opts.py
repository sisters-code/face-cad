import argparse
import os
import sys
import ref
from datetime import datetime
import numpy as np


class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-expID', default='resnet50', help='Experiment ID')
        self.parser.add_argument('-GPU', type=int, default=2, help='GPU id')
        self.parser.add_argument('-nThreads', type=int, default=8, help='nThreads')
        # self.parser.add_argument('-test', action='store_true', help='test phase')
        self.parser.add_argument('-loadModel', default='', help='Provide full path to a previously trained model')
        self.parser.add_argument('-arch', default='resnet50',
                                 help='network architecture')  # resnet50 resnet101 vgg16_bn

        self.parser.add_argument('-LR', type=float, default=0.01, help='Learning Rate')
        self.parser.add_argument('-dropLR', type=int, default=20, help='drop LR')
        self.parser.add_argument('-nEpochs', type=int, default=80, help='training epochs')
        self.parser.add_argument('-valIntervals', type=int, default=2, help='valid intervel')
        self.parser.add_argument('-trainBatch', type=int, default=32, help='Mini-batch size')
        # self.parser.add_argument('-task', default='ECG_CAD', help='ECG_CAD')
        self.parser.add_argument('-numBins', type=int, default=2, help='num of classes')
        self.parser.add_argument('-input_channels', type=int, default=3, help='The number of input channels, 3 | 12')

        self.parser.add_argument('-split_mode', type=str, default='assign', help='random | assign')
        # self.parser.add_argument('-dataset_suffix', type=str, default=datetime.now().strftime('%Y%m%d-%H%M%S'), help='')
        self.parser.add_argument('-trainBalance', default=True, action='store_true',
                                 help='Keep pos/neg ratio being 1 in training set')
        self.parser.add_argument('-Debug', type=int, default=0, help='Debug level')  # 0

    def parse(self):
        opt = self.parser.parse_args()

        # if opt.Debug:
        #     opt.nThreads = 1

        # if opt.task == 'ECG_CAD':
        #     opt.numOutput = 2

        # if opt.arch.startswith('vgg'):
        #     ref.inputRes = 224

        # if opt.test:
        #     opt.expID = opt.expID + 'TEST'
        # opt.saveDir = os.path.join(ref.expDir, opt.expID)

        opt.logDir = os.path.join(ref.expDir, opt.expID)

        # Generate candidate crop_window
        h_indices = (0, ref.oriRes - ref.inputRes)
        w_indices = (0, ref.oriRes - ref.inputRes)
        h_center_offset = (ref.oriRes - ref.inputRes) // 2
        w_center_offset = (ref.oriRes - ref.inputRes) // 2
        crop_idx = np.empty((5, 4), dtype=int)
        cnt = 0
        for i in h_indices:
            for j in w_indices:
                crop_idx[cnt, :] = (j, i, j + ref.inputRes, i + ref.inputRes)
                cnt += 1
        crop_idx[4, :] = (
            w_center_offset, h_center_offset, w_center_offset + ref.inputRes, h_center_offset + ref.inputRes)
        opt.cropIdx = crop_idx
        opt.val_auc_list = []
        opt.test_auc_list = []
        opt.best_val_auc = 0.
        opt.best_test_auc = 0.

        return opt
