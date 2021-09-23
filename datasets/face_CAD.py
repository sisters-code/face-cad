import torch.utils.data as data
import numpy as np
import ref
import os
import random
from PIL import Image, ImageDraw
import cv2
import json


class face_CAD(data.Dataset):
    def __init__(self, opt, split):
        print('==> Initializing {} data.'.format(split))
        annot = {}
        tags = ['class_id', 'imgPath']
        for tag in tags:
            annot[tag] = []

        read_data_pth = ref.read_data_pth
        train_pth = os.path.join(read_data_pth, 'train.txt')
        val_pth = os.path.join(read_data_pth, 'val.txt')
        test_pth = os.path.join(read_data_pth, 'test.txt')

        train_file = os.path.join(opt.logDir, 'train.txt')
        val_file = os.path.join(opt.logDir, 'val.txt')
        test_file = os.path.join(opt.logDir, 'test.txt')
        #把实验中使用的样本记录进logDir中的txt文件中
        if not (os.path.exists(train_file) or os.path.exists(val_file) or os.path.exists(test_file)):
            if opt.split_mode == 'assign':
                with open(train_pth, 'r') as train_pth, open(val_pth, 'r') as val_pth, open(test_pth, 'r') as test_pth, \
                        open(train_file, 'a+') as f_tr, open(val_file, 'a+') as f_val, open(test_file, 'a+') as f_test:
                    lines = train_pth.readlines()
                    random.shuffle(lines)
                    count = len(lines)
                    for idx, line in enumerate(lines):
                        # if line.strip():
                        f_tr.write(line)
                    line1s = test_pth.readlines()
                    for line1 in line1s:
                        f_test.write(line1)
                    line2s = val_pth.readlines()
                    for line2 in line2s:
                        f_val.write(line2)
                '''
            elif opt.split_mode == 'random':
                with open(train_pth, 'r') as total_pth, \
                        open(train_file, 'a+') as f_tr, open(val_file, 'a+') as f_val, open(test_file, 'a+') as f_test:
                    lines = total_pth.readlines()
                    random.shuffle(lines)  #
                    count = len(lines)
                    for idx, line in enumerate(lines):
                        if line.strip():
                            rnd = random.random()
                            if rnd < 0.1:
                                f_val.write(line)
                            elif rnd < 0.2:
                                f_test.write(line)
                            else:
                                f_tr.write(line)
            elif opt.split_mode == "assign_noVal":
                with open(train_pth, 'r') as train_pth, open(test_pth, 'r') as test_pth, \
                        open(train_file, 'a+') as f_tr, open(val_file, 'a+') as f_val, open(test_file, 'a+') as f_test:
                    lines = train_pth.readlines()
                    random.shuffle(lines)
                    count = len(lines)
                    for idx, line in enumerate(lines):
                        if line.strip():
                            f_tr.write(line)

                    line1s = test_pth.readlines()
                    for line1 in line1s:
                        f_test.write(line1)
                    line2s = val_pth.readlines()
                    for line2 in line2s:
                        f_val.write(line2)
            '''
            else:
                raise NotImplementedError
        elif (os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file)):
            pass
        else:
            raise Exception('dataset is incompleted!')

        # load dataset
        if split == 'train':
            f_r = open(train_file, 'r')
        elif split == 'val':
            f_r = open(val_file, 'r')
        elif split == 'test':
            f_r = open(test_file, 'r')
        else:
            raise Exception('Wrong split!')

        #trainBalance为true，则将正负样本中数量少的部分自我复制加上，使样本数量平衡
        if split == 'train' and opt.trainBalance:
            # make sample balance
            pos_img_paths = []
            pos_labels = []
            neg_img_paths = []
            neg_labels = []
            for line in f_r.readlines():
                items = line.strip().split('+')
                if len(items) == 2:
                    if int(items[1]) == 1:
                        pos_img_paths.append(items[0])
                        pos_labels.append(int(items[1]))
                    elif int(items[1]) == 0:
                        neg_img_paths.append(items[0])
                        neg_labels.append(int(items[1]))
                    else:
                        pass
            pos_size = len(pos_labels)
            neg_size = len(neg_labels)
            if pos_size > neg_size:
                size_extra = pos_size - neg_size
                neg_idx_extra = []
                for i in range(size_extra // neg_size):
                    neg_idx_extra += [i for i in range(neg_size)]
                neg_idx_extra += random.sample(range(neg_size), size_extra % neg_size)

                neg_img_paths = neg_img_paths + list(np.asarray(neg_img_paths)[neg_idx_extra])
                neg_labels = neg_labels + list(np.asarray(neg_labels)[neg_idx_extra])
            elif pos_size < neg_size:
                size_extra = neg_size - pos_size
                pos_idx_extra = []
                for i in range(size_extra // pos_size):
                    pos_idx_extra += [i for i in range(pos_size)]
                pos_idx_extra += random.sample(range(pos_size), size_extra % pos_size)
                pos_img_paths = pos_img_paths + list(np.asarray(pos_img_paths)[pos_idx_extra])
                pos_labels = pos_labels + list(np.asarray(pos_labels)[pos_idx_extra])
            else:
                pass
            annot['imgPath'].extend(pos_img_paths + neg_img_paths)
            annot['class_id'].extend(pos_labels + neg_labels)
        else:
            for line in f_r.readlines():
                items = line.strip().split('+')
                if len(items) == 2:
                    annot['imgPath'].append(items[0])
                    annot['class_id'].append(int(items[1]))
        f_r.close()

        for tag in tags:
            annot[tag] = np.asarray(annot[tag])

        annot['index'] = np.arange(len(annot['class_id']))
        tags = tags + ['index']
        self.split = split
        self.opt = opt
        self.annot = annot
        self.nSamples = len(annot['class_id'])
        self.pos_num = np.sum(annot['class_id'] == 1)
        self.neg_num = np.sum(annot['class_id'] == 0)
        self.input_channels = self.opt.input_channels
        print('pos/neg ratio is {}/{}={}'.format(self.pos_num, self.neg_num, self.pos_num * 1.0 / self.neg_num))
        print('Loaded {} {} samples'.format(split, self.nSamples))

    #输入人脸的灰度图
    def LoadImage_1channels(self, index):
        img_file = self.annot['imgPath'][index]
        img_pil = Image.open(img_file).convert('L')
        img_pil = img_pil.resize((ref.oriRes, ref.oriRes), Image.ANTIALIAS)

        if self.split == 'train':
            # data augmentation
            rnd = np.random.rand()
            crop_idx = np.random.randint(len(self.opt.cropIdx))
            img_crop = img_pil.crop(self.opt.cropIdx[crop_idx])
            if rnd > 0.5:
                img_crop = img_crop.transpose(Image.FLIP_LEFT_RIGHT)
            # add rotation augmentation
            rotate_angle = (np.random.rand() * 2 - 1) * 30  # rotation range: [-30, 30)
            img_crop = img_crop.rotate(rotate_angle)
        else:
            img_crop = img_pil.crop(self.opt.cropIdx[-1])
        img_crop = np.array(img_crop)
        #创建CLAHE对象，用于提升对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # 限制对比度的自适应阈值均衡化
        img_crop = clahe.apply(img_crop)
        # 使用全局直方图均衡化
        img_crop = cv2.equalizeHist(img_crop)

        img_crop = np.expand_dims(img_crop, axis = 2)
        img_array = img_crop.transpose(2, 0, 1).astype(np.float32) / 256.
        return img_array

    def LoadImage_3channels(self, index):
        img_file = self.annot['imgPath'][index]
        # img_list = ['1.JPG']
        # img_idx = np.random.randint(len(img_list))
        # img_file = os.path.join(img_file, img_list[img_idx])
        img_pil = Image.open(img_file)
        img_pil = img_pil.resize((ref.oriRes, ref.oriRes), Image.ANTIALIAS)

        if self.split == 'train':
            # data augmentation
            rnd = np.random.rand()
            crop_idx = np.random.randint(len(self.opt.cropIdx))
            img_crop = img_pil.crop(self.opt.cropIdx[crop_idx])
            if rnd > 0.5:
                img_crop = img_crop.transpose(Image.FLIP_LEFT_RIGHT)
            # add rotation augmentation
            rotate_angle = (np.random.rand() * 2 - 1) * 30  # rotation range: [-30, 30)
            img_crop = img_crop.rotate(rotate_angle)
        else:
            img_crop = img_pil.crop(self.opt.cropIdx[-1])
        img_crop = np.array(img_crop)
        img_array = img_crop.transpose(2, 0, 1).astype(np.float32) / 256.
        return img_array

    def LoadImage(self, index):
        img_path = self.annot['imgPath'][index]
        # img_list = ['1.JPG'] # VGG
        img_list = ['1.JPG', '2.JPG', '3.JPG', '4.JPG']
        img_read = []
        for img_name in img_list:
            img_file = os.path.join(img_path, img_name)
            if not os.path.exists(img_file):
                img_file = os.path.join(img_path, img_name.lower())
                if not os.path.exists(img_file):
                    raise Exception('Image not found: {}'.format(img_file))
            img_pil = Image.open(img_file)
            img_pil = img_pil.resize((ref.oriRes, ref.oriRes), Image.ANTIALIAS)

            if self.split == 'train':
                # data augmentation
                rnd = np.random.rand()
                crop_idx = np.random.randint(len(self.opt.cropIdx))
                img_crop = img_pil.crop(self.opt.cropIdx[crop_idx])
                if rnd > 0.5:
                    img_crop = img_crop.transpose(Image.FLIP_LEFT_RIGHT)
                # add rotation augmentation
                rotate_rnd = np.random.rand()
                if rotate_rnd > 0.5:
                    rotate_angle = (np.random.rand() * 2 - 1) * 30  # rotation range: [-30, 30)
                    img_crop = img_crop.rotate(rotate_angle)
                    img_read.append(np.array(img_crop))

            else:
                img_crop = img_pil.crop(self.opt.cropIdx[-1])
                img_read.append(np.array(img_crop))

        img_array = np.concatenate(img_read, axis=2).transpose(2, 0, 1).astype(np.float32) / 256.
        return img_array

    def __getitem__(self, index):
        if self.input_channels == 1:
            inp = self.LoadImage_1channels(index)
        elif self.input_channels == 3:
            inp = self.LoadImage_3channels(index)
        elif self.input_channels == 12:
            inp = self.LoadImage(index)
        else:
            raise ValueError("Input channels should be 3 or 12!")

        class_id = self.annot['class_id'][index]

        return inp, class_id, self.annot['imgPath'][index]

    def __len__(self):
        return self.nSamples
