#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Data_Loader
======
A class for something.
"""

import datetime
import random
import sys
sys.path.insert('.')
sys.path.insert('..')
from sequence_labeling.data_processor import DataProcessor


class Data_Loader(Data_Processor):
    def __init__(self):
        super(Data_Loader, self).__init__()
        print('Data_Loader Class Init...')

    # 依据词典、标签索引将input、output数据转化为向量
    # 参数op_mode的值为“train”、“val”或“test”
    def data_generator(self, batch_size=64, shuffle=True, op_mode=None):
        input_path = self.data_root + self.inputdata_file_path + '.' + op_mode
        output_path = self.data_root + self.outputdata_file_path + '.' + op_mode
        word_dict = Data_Processor().load_vocab()
        tags_dict = Data_Processor().load_tags()

        print("Load input data from {}.".format(input_path))
        print("Load output data from {}.".format(output_path))
        with open(input_path, 'r') as fp:
            input_data = fp.readlines()

        with open(output_path, 'r') as fp:
            output_data = fp.readlines()

        if shuffle:
            data = list(zip(input_data, output_data))
            random.shuffle(data)
            input_data, output_data = zip(*data)

        for i in range(0, len(output_data), batch_size):
            batch_input = []
            batch_output = []
            for line in input_data[i: i + batch_size]:    # 依据词典，将input转化为向量
                new_line = []
                for word in line.strip().split():
                    new_line.append(int(word_dict[word]) if word in word_dict else int(word_dict['UNK']))
                batch_input.append(new_line)
            for line in output_data[i: i + batch_size]:    # 依据Tag词典，将output转化为向量
                new_line = []
                for tag in line.strip().split():
                    new_line.append(int(tags_dict[tag]) if tag in tags_dict else print("There is a wrong Tag in {}!".format(line)))
                batch_output.append(new_line)
            yield batch_input, batch_output


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    loader = Data_Loader()
    for sentence, tag in loader.data_generator(op_mode='train'):
        print(sentence)
        print(tag)

    end_time = datetime.datetime.now()
    print('Data_Loader takes {} seconds.'.format((end_time - start_time).seconds))
