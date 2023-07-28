import os
import numpy as np
import torch
import math
import pandas as pd
import functools
import random
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import cv2
import random
import h5py
from matplotlib import pyplot as plt
from utils.timefeatures import time_features


class MatrixDataset(Dataset):
    large_dataset_flag = False

    def __init__(self, path, all_batch_num, fn_get_traffic_matrix=None, predict_matrix_num=3, input_matrix_num=4
                 , gpu_mode=False, sampling_rate1=None, sampling_rate2=None, scaler=1, using_tube_samping=False):
        assert sampling_rate2 is None or (sampling_rate2 > 0 and sampling_rate2 <= 1)
        assert sampling_rate1 is None or (sampling_rate1 > 0 and sampling_rate1 <= 1)

        self.all_batch_num = all_batch_num
        self.predict_matrix_num = predict_matrix_num
        self.input_matrix_num = input_matrix_num
        self.gpu_mode = gpu_mode
        self.sampling_rate1 = sampling_rate1  # sampling rate of matrix 即空间采样率 SSR
        self.sampling_rate2 = sampling_rate2  # sampling rate of time interval 丢弃率Discard Rate
        self.scaler = scaler
        self.using_tube_samping = using_tube_samping

        if fn_get_traffic_matrix is None:  # get data matrix fn
            fn_get_traffic_matrix = get_traffic_matrix_abilene

        all_need_matrix_num = (all_batch_num+self.input_matrix_num)*(1+self.predict_matrix_num)
        if sampling_rate2 is not None and sampling_rate2 < 1:
            all_need_matrix_num = int(all_need_matrix_num/sampling_rate2)
        # get raw data
        tms, time_seq = fn_get_traffic_matrix(path, all_need_matrix_num)
        # PE2
        time_seq = pd.to_datetime(time_seq)
        time_seq -= pd.to_datetime('Jul  1 1900, 00:00:00')
        time_seq = pd.Series(time_seq).dt.total_seconds()
        time_seq = torch.tensor(time_seq).numpy()

        self.time_seq = time_seq
        self.tms = tms if type(tms) is np.ndarray else torch.from_numpy(tms)

        # normalize un-missing data
        # self.minone, self._range, self.tms = self.__normalization__(self.tms)
        self.tms = torch.from_numpy(self.tms)*self.scaler
        self.time_seq = torch.from_numpy(self.time_seq)

        if self.sampling_rate2 is not None and self.sampling_rate2 < 1:
            # should delete some matrix
            indices = list(range(len(self.tms)))
            random.shuffle(indices)  # 打乱序号
            indices = indices[:int(self.sampling_rate2*len(self.tms))]  # 截去多余tms
            indices.sort()
            self.tms = self.tms[indices]
            self.time_seq = self.time_seq[indices]

        print(f'sampling_rate1={sampling_rate1}\tsampling_rate2={sampling_rate2}')
        print(f'max={self.tms.max()} min={self.tms.min()} mean={self.tms.mean()} shape={self.tms.shape}')
        if sampling_rate1 is not None:
            self.masks = self.get_masks(self.tms, using_tube_samping)
        if gpu_mode:
            self.large_dataset_flag = False
            self.tms = self.tms.to(torch.device('cuda:0'))
            self.time_seq = self.time_seq.to(torch.device('cuda:0'))
            self.masks = self.masks.to(torch.device('cuda:0'))

    def get_masks(self, tms, using_tube_sampling=False):
        masks = []
        if not using_tube_sampling:
            for i in tms:
                masks.append(self.produce_a_mask(i))
        else:
            tube_mask = self.produce_a_mask(tms[0])
            masks = [tube_mask for i in tms]
        masks = torch.stack(masks)
        masks = masks.type(torch.FloatTensor)

        if self.gpu_mode and self.large_dataset_flag:
            masks = masks.to(torch.device('cuda:0'))
        return masks

    def produce_a_mask(self, matrix):
        amount = int(torch.prod(torch.from_numpy(np.array(matrix.shape))))
        one_num = int(amount * self.sampling_rate1)

        order = list(range(amount))
        random.shuffle(order)
        idx = order[:one_num]
        mask = np.zeros(amount)
        mask[idx] = 1
        return torch.from_numpy(mask.reshape(matrix.shape))

    def __getitem__(self, index):
        # train data
        head_matrix_pos = index*(1+self.predict_matrix_num)
        tail_matrix_pos = (index+self.input_matrix_num)*(1+self.predict_matrix_num)

        indices = list(range(head_matrix_pos, tail_matrix_pos, self.predict_matrix_num+1))  # 均匀取值
        train = self.tms[indices]
        time_seq = self.time_seq[indices]

        if self.sampling_rate1 is not None and self.sampling_rate1!=1:  # partial sampling
            train_mask = self.masks[head_matrix_pos: tail_matrix_pos: self.predict_matrix_num+1]
            train = train * train_mask

        # valid
        head_valid_pos = (index+self.input_matrix_num//2-1)*(1+self.predict_matrix_num)+1
        tail_valid_pos = (index+self.input_matrix_num//2-1)*(1+self.predict_matrix_num)+1+self.predict_matrix_num
        valid = self.tms[head_valid_pos-1: tail_valid_pos]  # the input frame at [head_valid_pos-1] also should be recovered
        target_times = self.time_seq[head_valid_pos-1: tail_valid_pos]

        train = torch.unsqueeze(train, dim=0).to(torch.float32)
        valid = torch.unsqueeze(valid, dim=0).to(torch.float32)

        if self.gpu_mode and self.large_dataset_flag:
            # Cause hard to sending full dataset into GPU
            train = train.to(torch.device('cuda:0'))
            valid = valid.to(torch.device('cuda:0'))
            time_seq = time_seq.to(torch.device('cuda:0'))
            target_times = target_times.to(torch.device('cuda:0'))
        return [train, valid, time_seq, target_times]

    def __len__(self):
        return self.all_batch_num

    def __normalization__(self, datalist):
        """ Normalize a batch data """
        # if type(datalist) is list:
        #     datalist = np.array(datalist)
        # if type(datalist) is np.ndarray:
        #     datalist = torch.from_numpy(datalist)
        # # Batch Norm
        # maxone, minone = datalist.max(), datalist.min()
        # _range = maxone - minone
        # temp = (datalist - minone) / _range
        # return minone, _range, temp

        # Instance Norm
        if type(datalist) is torch.Tensor:
            datalist = datalist.numpy()
        maxone, minone = datalist.max(-1).max(-1), datalist.min(-1).min(-1)
        maxone, minone = maxone.reshape(-1, 1, 1), minone.reshape(-1, 1, 1)
        _range = maxone - minone

        temp = (datalist - minone) / _range
        return minone, _range, torch.from_numpy(temp)


def get_traffic_matrix_geant(path: str = '.', all_batch_size=1000):
    files = os.listdir(path)  # list current path files
    # filter other file
    index = len(files) - 1
    while index >= 0:
        if files[index].find('IntraTM') == -1:
            del (files[index])
        index -= 1
    # sort file
    files.sort()  # 时间从前向后排序

    assert len(files) >= all_batch_size
    files = files[:all_batch_size]
    tms, time_stamps = [], []  # traffic matrix, timestamp

    print('Begin load GEANT')
    # 解析xml file
    tree = ET.ElementTree()
    for timestamp in files:
        tm = [[0.0 for j in range(24)] for i in range(24)]  # 紧接着下面去除了[?][0]及[0][?]
        ele = tree.parse(path + '/' + timestamp)
        for row in ele[1]:
            row_id = int(row.get('id'))
            for node in row:
                col_id = int(node.get('id'))
                tm[row_id][col_id] = float(node.text)
        tms.append(tm)
        stamp_str = timestamp[8:-4]  # 2005-01-01-00-30
        stamp_str = stamp_str[:-5] + stamp_str[-5:].replace('-', ':') + ':00'  # format: '2005-01-01-00:30:00'
        time_stamps.append(stamp_str)
    print('GEANT loaded')

    # 去除读入时，行列index=0时的边缘
    temp = np.array(tms)  # [all_batch, 24, 24] 分别为 batch，行，列
    x, y = np.split(temp, [1], 1)  # 切除行
    x, tms = np.split(y, [1], 2)  # 切除列
    time_stamps = np.array(time_stamps)

    # tms = np.concatenate([tms, tms], axis=-1)  # 尺寸拼接
    # tms = np.concatenate([tms, tms], axis=-2)

    tms = np.clip(tms, 0, 2e6) # filter anomaly value
    return tms, time_stamps


def get_traffic_matrix_abilene(path: str = '.', all_batch_size=1000):
    files = os.listdir(path)  # list current path files
    # filter other file
    index = len(files) - 1
    while index >= 0:
        if files[index].find('tm.2004') == -1:
            del (files[index])
        index -= 1

    # sort file
    files.sort()  # 时间从前向后排序

    assert len(files) >= all_batch_size
    files = files[:all_batch_size]
    tms, time_stamps = [], []  # traffic matrix, timestamp

    print('Begin load Abilene')
    for timestamp in files:
        tm = []
        with open(path + '/' + timestamp) as file:
            while True:
                line = file.readline()
                if line == '':
                    break
                if line[0] == '#':
                    continue
                line = line.strip().split(',')
                tm.append([float(num) for num in line])
        tms.append(tm)
        stamp_str = timestamp[3:-4].replace('.', ' ')
        stamp_str = stamp_str[:-8] + stamp_str[-8:].replace('-', ':')  # format: '2004-03-01 00:00:00'
        time_stamps.append(stamp_str)
    print('Data loaded')

    tms = np.array(tms)  # [all_batch, 12, 12] 分别为 batch，行，列
    time_stamps = np.array(time_stamps)
    return tms, time_stamps


if __name__ == '__main__':
    pass


