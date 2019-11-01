# =====================
# Adaptive Graph Guided Embedding for Multi-label Annotation (AG2E)
# =====================
# Author: Lichen Wang, Yunyu Liu
# Date: Mar, 2019
# E-mail: wanglichenxj@gmail.com, liu.yuny@husky.neu.edu

# @inproceedings{GMVAR_lichen,
#   title={Generative Multi-View Human Action Recognition},
#   author={Wang, Lichen and Ding, Zhengming and Tao, Zhiqiang and Liu, Yunyu and Fu, Yun},
#   booktitle={Proc. IEEE International Conference on Computer Vision},
#   year={2019},
#   organization={}
# }
# =====================

import random

# =====================
# load the training and testing sample features
# The feature is extracted by existing approach without fine-tuning
# Input: Dataset name
# Output: training/testing RGB/depth feature/label of the data
# =====================
class data_loader:

    def __init__(self, database_name='UCB'):

        self.filename = database_name
        self.data_x = [] # training and testing RGB + depth feature
        self.data_label = [] # training and testing depth label
        self.train_data_x = [] # training depth feature
        self.train_data_y = [] # training RGB feature
        self.train_data_label = [] # training label
        self.test_data_x = [] # testing depth feature
        self.test_data_y = [] # testing RGB feature
        self.test_data_label = [] # testing label
        self.train_data_xy = [] # training RGB + depth feature
        self.test_data_xy = [] # testing RGB + depth feature

    def read_train(self):

        # Depth feature -> 110-dimension
        # RGB feature -> 3x2048 dimension
        feature_num1 = 110
        feature_num2 = 3*2048
        feature_num = feature_num1 + feature_num2
        num = 0

        # load .csv file for training
        f = open('action_data/' + self.filename+'_total_train.csv', 'r')
        for i in f:
            num += 1
            row1 = i.rstrip().split(',')[:-1]
            row = [float(x) for x in row1]
            self.data_x.append(row[0:feature_num])
            self.data_label.append(row[feature_num:])
            self.train_data_x.append(row[0:feature_num1])
            self.train_data_y.append(row[feature_num1:feature_num1+feature_num2])
            self.train_data_xy.append(row[0:feature_num1 + feature_num2])
            self.train_data_label.append(row[feature_num1+feature_num2:])
        f.close()

        # load .csv file for training
        f = open('action_data/' + self.filename+'_total_test.csv', 'r')
        for i in f:
            num += 1
            row1 = i.rstrip().split(',')[:-1]
            row = [float(x) for x in row1]
            self.data_x.append(row[0:feature_num])
            self.data_label.append(row[feature_num:])
            self.test_data_x.append(row[0:feature_num1])
            self.test_data_y.append(row[feature_num1:feature_num1 + feature_num2])
            self.test_data_xy.append(row[0:feature_num1 + feature_num2])
            self.test_data_label.append(row[feature_num1 + feature_num2:])
        f.close()

        # got the sample number
        self.sample_total_num = len(self.data_x)
        self.sample_train_num = len(self.train_data_x)
        self.sample_test_num = len(self.test_data_x)
        print(self.sample_total_num)

    # randomly choose _batch_size RGB and depth feature in the training set
    def train_next_batch(self, _batch_size):
        xx = [] # training batch of depth features
        yy = [] # training batch of RGB features
        zz = [] # training batch of labels
        for sample_num in random.sample(range(self.sample_train_num), _batch_size):
            xx.append(self.train_data_x[sample_num])
            yy.append(self.train_data_y[sample_num])
            zz.append(self.train_data_label[sample_num])
        return yy, xx, zz

    # randomly choose _batch_size RGB and depth feature in the testing set
    def test_next_batch(self, _batch_size):
        xx = [] # testing batch of depth features
        yy = [] # testing batch of RGB features
        zz = [] # testing batch of labels
        for sample_num in random.sample(range(self.sample_test_num), _batch_size):
            xx.append(self.test_data_x[sample_num])
            yy.append(self.test_data_y[sample_num])
            zz.append(self.test_data_label[sample_num])
        return yy, xx, zz

