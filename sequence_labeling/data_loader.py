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
# from keras.preprocessing.sequence import pad_sequences
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

    # 参数run_mode的值为“train”、“eval”或“test”
    def data_generator(self, data_path, run_mode):
        input_data_path = '{}{}.{}'.format(data_path, self.inputdata_file_name, run_mode)
        output_data_path = '{}{}.{}'.format(data_path, self.outputdata_file_name, run_mode)
        word_dict = DataProcessor(**self.config).load_vocab()
        tags_dict = DataProcessor(**self.config).load_tags()

        # print("Load input data from {}.".format(input_path))
        # print("Load output data from {}.".format(output_path))
        with open(input_data_path, 'r', encoding='utf-8') as fp:
            input_data = fp.readlines()

        with open(output_data_path, 'r', encoding='utf-8') as fp:
            output_data = fp.readlines()

        if self.shuffle:
            data = list(zip(input_data, output_data))
            random.shuffle(data)
            input_data, output_data = zip(*data)

        for i in range(0, len(output_data), self.batch_size):
            # 构建batch数据
            batch_input = input_data[i: i + self.batch_size]
            batch_output = output_data[i: i + self.batch_size]

            yield batch_input, batch_output


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    config_path = './config/cmed/dl/cmed.dl.base_model.norm.json'
    if not os.path.exists(config_path):
        raise RuntimeError("There is no {} config.".format(config_path))
    config = json.load(open(config_path, 'r', encoding='utf-8'))
    print('config: ', config)

    data_path = '{}'.format(config['data_root'])

    loader = DataLoader(**config)
    for batch_x_padded, x_lengths_original, batch_y_padded, y_lengths_original in loader.data_generator(
            data_path=data_path, run_mode='train'):
        print(batch_x_padded)
        print(batch_y_padded)

    end_time = datetime.datetime.now()
    print('Data_Loader takes {} seconds.'.format((end_time - start_time).seconds))
