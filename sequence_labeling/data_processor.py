#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
DataProcessor
======
A class for something.
"""

import argparse
import datetime
import json
import os
import random

import nltk


class DataProcessor(object):
    def __init__(self):
        print('Data_Processor Class Init...')
        self.data_root = './datasets/cmed/'
        self.inputdata_file_path = 'data.input'
        self.outputdata_file_path = 'data.output'
        self.dict_file_path = 'vocab.json'
        self.tags_file_path = 'tags.json'

    # 将数据存入文件
    def save(self, data, filepath):
        count = 0
        with open(filepath, 'w') as fw:
            for line in data:
                fw.write(line + '\n')
                count += 1
        print("Done for saving {} lines to {}.".format(count, filepath))

    # 将data.input和data.output划分成训练集、验证集和测试集
    def split_data(self, split_rate=0.7, *args, **kwargs):
        with open(self.data_root + self.inputdata_file_path, 'r') as fp:
            data_input = list(map(lambda x: x.strip(), fp.readlines()))

        with open(self.data_root + self.outputdata_file_path, 'r') as fp:
            data_output = list(map(lambda x: x.strip(), fp.readlines()))

        random.seed(10)
        data = list(zip(data_input, data_output))
        random.shuffle(data)
        data_input, data_output = zip(*data)

        data_size = len(data_output)
        train_data_input = data_input[:int(data_size * split_rate)]
        train_data_output = data_output[:int(data_size * split_rate)]
        val_data_input = data_input[
                         int(data_size * split_rate): int(data_size * (split_rate + (1 - split_rate) / 2))]
        val_data_output = data_output[
                          int(data_size * split_rate): int(data_size * (split_rate + (1 - split_rate) / 2))]
        test_data_input = data_input[int(data_size * (split_rate + (1 - split_rate) / 2)):]
        test_data_output = data_output[int(data_size * (split_rate + (1 - split_rate) / 2)):]

        train_inputdata_path = self.data_root + self.inputdata_file_path + '.train'
        train_outputdata_path = self.data_root + self.outputdata_file_path + '.train'
        val_inputdata_path = self.data_root + self.inputdata_file_path + '.val'
        val_outputdata_path = self.data_root + self.outputdata_file_path + '.val'
        test_inputdata_path = self.data_root + self.inputdata_file_path + '.test'
        test_outputdata_path = self.data_root + self.outputdata_file_path + '.test'

        self.save(train_data_input, train_inputdata_path)
        self.save(train_data_output, train_outputdata_path)
        self.save(val_data_input, val_inputdata_path)
        self.save(val_data_output, val_outputdata_path)
        self.save(test_data_input, test_inputdata_path)
        self.save(test_data_output, test_outputdata_path)

    # 依据训练数据构建词索引词典(word:id)
    def get_vocab(self, cover_rate=1, mincount=0, *args, **kwargs):
        data_input_path = self.data_root + self.inputdata_file_path

        word_count_dict = {}
        total_word_count = 0
        with open(data_input_path, 'r') as fp:
            for line in fp.readlines():
                for word in line.strip().split():
                    total_word_count += 1
                    if word not in word_count_dict:
                        word_count_dict[word] = 1
                    else:
                        word_count_dict[word] += 1
        sorted_word_count_dict = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
        print("There are {} words originally.".format(len(sorted_word_count_dict)))

        word_dict = {'UNK': 0, 'SOS': 1, 'EOS': 2}
        tmp_word_count = 0
        for word, count in sorted_word_count_dict:
            print("{}:{}".format(word, count))
            tmp_word_count += count
            current_rate = tmp_word_count / total_word_count
            if count > mincount and current_rate <= cover_rate:
                word_dict[word] = len(word_dict)
        print("There are {} words finally.".format(len(word_dict)))

        word_dict_path = self.data_root + self.dict_file_path
        with open(word_dict_path, 'w') as fw:
            json.dump(word_dict, fw)
        print("Successfully save word dict to {}.".format(word_dict_path))

    # 读取vocab.json的内容，返回值为{}类型。
    def load_vocab(self):
        word_dict_path = self.data_root + self.dict_file_path
        with open(word_dict_path, 'r') as fp:
            word_dict = json.load(fp)
            print("Load word dict from {}.".format(word_dict_path))
            return word_dict

    # 读取tags.json的内容，返回值为{}类型。
    def load_tags(self):
        tagfile_path = self.data_root + self.tags_file_path
        with open(tagfile_path, 'r') as tfp:
            tags_dict = json.load(tfp)
            print("Load Tags dict from {}.".format(tagfile_path))
            return tags_dict


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    # 生成词典vocab.json
    data_processor = DataProcessor()
    data_processor.get_vocab()
    print(data_processor.load_vocab())

    # 将data.input和data.output划分成训练集、验证集和测试集
    data_processor.split_data()

    # print(data_processor.load_tags())

    end_time = datetime.datetime.now()
    print('dataProcessor takes {} seconds.'.format((end_time - start_time).seconds))
