# -*- coding: utf-8 -*-
import os
import math
import codecs
import random
import numpy as np
from glob import glob
from PIL import Image

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.utils import np_utils, Sequence
from sklearn.model_selection import StratifiedShuffleSplit


class BaseSequence(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch
    BaseSequence可直接用于fit_generator的generator参数
    fit_generator会将BaseSequence再次封装为一个多进程的数据流生成器
    而且能保证在多进程下的一个epoch中不会重复取相同的样本
    """
    def __init__(self, img_paths, labels, batch_size, img_size):
        assert len(img_paths) == len(labels), "len(img_paths) must equal to len(lables)"
        assert img_size[0] == img_size[1], "img_size[0] must equal to img_size[1]"
        self.x_y = np.hstack((np.array(img_paths).reshape(len(img_paths), 1), np.array(labels)))
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)

    def preprocess_img(self, img_path):
        """
        image preprocessing
        you can add your special preprocess method here
        """
        img = image.load_img(img_path, target_size=(self.img_size[0], self.img_size[1]))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        return x

    def __getitem__(self, idx):
        batch_x = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 0]
        batch_y = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 1:]
        batch_x = np.array([self.preprocess_img(img_path) for img_path in batch_x])
        batch_y = np.array(batch_y).astype(np.float32)
        return batch_x, batch_y

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        np.random.shuffle(self.x_y)


def data_flow(train_data_dir, batch_size, num_classes, input_size):
    label_files = glob(os.path.join(train_data_dir, '*.txt'))
    random.shuffle(label_files)
    img_paths = []
    labels = []
    for index, file_path in enumerate(label_files):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_path))
            continue
        img_name = line_split[0]
        label = int(line_split[1])
        img_paths.append(os.path.join(train_data_dir, img_name))
        labels.append(label)
    img_paths = np.array(img_paths)
    labels = np_utils.to_categorical(labels, num_classes)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=500, random_state=0)  # 您可以根据自己的需要调整 test_size 的大小
    sps = sss.split(img_paths, labels)
    for sp in sps:
        train_index, validation_index = sp
    print('total samples: %d, training samples: %d, validation samples: %d'
          % (len(img_paths), len(train_index), len(validation_index)))
    train_img_paths = img_paths[train_index]
    validation_img_paths = img_paths[validation_index]
    train_labels = labels[train_index]
    validation_labels = labels[validation_index]

    train_sequence = BaseSequence(train_img_paths, train_labels, batch_size, [input_size, input_size])
    validation_sequence = BaseSequence(validation_img_paths, validation_labels, batch_size, [input_size, input_size])

    return train_sequence, validation_sequence
