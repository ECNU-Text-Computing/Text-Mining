#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Data_Loader
======
A class for something.
"""

import datetime
import random
from data_processor import DataProcessor
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import json


class DataLoader:
    def __init__(self, **config):
        super(DataLoader, self).__init__()
        # print('Data_Loader Class Init...')
        self.config = config
        self.data_root = config['data_root']
        self.inputdata_file_name = config['inputdata_file_name']
        self.outputdata_file_name = config['outputdata_file_name']
        self.n_folds = config['n_folds']
        self.shuffle = config['shuffle']
        self.batch_size = config['batch_size']

        self.pad_token = config['pad_token']
        vocab = DataProcessor(**self.config).load_vocab()
        self.padding_idx = vocab[self.pad_token]

    # 依据词典、标签索引将input、output数据转化为向量
    # 参数run_mode的值为“train”、“val”或“test”
    def data_generator(self, data_path, run_mode):
        input_data_path = '{}{}.{}'.format(data_path, self.inputdata_file_name, run_mode)
        output_data_path = '{}{}.{}'.format(data_path, self.outputdata_file_name, run_mode)
        word_dict = DataProcessor(**self.config).load_vocab()
        tags_dict = DataProcessor(**self.config).load_tags()

        # print("Load input data from {}.".format(input_path))
        # print("Load output data from {}.".format(output_path))
        with open(input_data_path, 'r') as fp:
            input_data = fp.readlines()

        with open(output_data_path, 'r') as fp:
            output_data = fp.readlines()

        if self.shuffle:
            data = list(zip(input_data, output_data))
            random.shuffle(data)
            input_data, output_data = zip(*data)

        for i in range(0, len(output_data), self.batch_size):
            # 构建batch数据，并向量化
            batch_input = []
            batch_output = []
            for line in input_data[i: i + self.batch_size]:  # 依据词典，将input转化为向量
                new_line = []
                for word in line.strip().split():
                    new_line.append(int(word_dict[word]) if word in word_dict else int(word_dict['UNK']))
                batch_input.append(new_line)
            for line in output_data[i: i + self.batch_size]:  # 依据Tag词典，将output转化为向量
                new_line = []
                for tag in line.strip().split():
                    new_line.append(
                        int(tags_dict[tag]) if tag in tags_dict else print("There is a wrong Tag in {}!".format(line)))
                batch_output.append(new_line)

            # 为了满足pack_padded_sequence的要求，将batch数据按input数据的长度排序
            batch_data = list(zip(batch_input, batch_output))
            batch_input_data, batch_output_data = zip(*batch_data)
            in_out_pairs = []  # [[batch_input_data_item, batch_output_data_item], ...]
            for n in range(len(batch_input_data)):
                new_pair = []
                new_pair.append(batch_input_data[n])
                new_pair.append(batch_output_data[n])
                in_out_pairs.append(new_pair)
            in_out_pairs = sorted(in_out_pairs, key=lambda x: len(x[0]), reverse=True)  # 根据每对数据中第1个数据的长度排序

            # 排序后的数据重新组织成batch_x, batch_y
            batch_x = []
            batch_y = []
            for each_pair in in_out_pairs:
                batch_x.append(each_pair[0])
                batch_y.append(each_pair[1])

            # 按最长的数据pad
            batch_x_padded, x_lengths_original = self.pad_seq(batch_x)
            batch_y_padded, y_lengths_original = self.pad_seq(batch_y)

            yield batch_x_padded, x_lengths_original, batch_y_padded, y_lengths_original

    def pad_seq(self, X):
        X_lengths = [len(st) for st in X]
        # create an empty matrix with padding tokens
        pad_token = self.padding_idx
        longest_sent = max(X_lengths)
        batch_size = len(X)
        padded_X = np.ones((batch_size, longest_sent)) * pad_token
        # copy over the actual sequences
        for i, x_len in enumerate(X_lengths):
            sequence = X[i]
            padded_X[i, 0:x_len] = sequence[:x_len]
        return padded_X, X_lengths


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    config_path = './config/cmed/dl/cmed.dl.base_model.norm.json'
    if not os.path.exists(config_path):
        raise RuntimeError("There is no {} config.".format(config_path))
    config = json.load(open(config_path, 'r'))
    print('config: ', config)

    data_path = '{}'.format(config['data_root'])

    loader = DataLoader(**config)
    for batch_x_padded, x_lengths_original, batch_y_padded, y_lengths_original in loader.data_generator(data_path= data_path, run_mode='train'):
        print(batch_x_padded)
        print(batch_y_padded)

    end_time = datetime.datetime.now()
    print('Data_Loader takes {} seconds.'.format((end_time - start_time).seconds))
