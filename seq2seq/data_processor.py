#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Data_Processor
======
A class for something.
"""

import os
import sys
import argparse
import datetime

import json
import random
import nltk


class DataProcessor(object):
    def __init__(self):
        print('Init...')
        # 数据集的地址。
        self.data_root = './datasets/'
        # 原始数据集存放的地址。
        self.original_root = self.data_root + 'original/'
        self.aapr_root = self.original_root + 'AAPR_Dataset/'

        # 实验中间结果存放的地址。
        self.exp_root = './exp/'
        # 日志存放的地址。
        self.log_root = './logs/'

        # 如果没有这两个地址，下面的代码将会创建一个。
        if not os.path.exists(self.exp_root):
            os.mkdir(self.exp_root)

        # 如果没有这两个地址，下面的代码将会创建一个。
        if not os.path.exists(self.log_root):
            os.mkdir(self.log_root)

    ###################################
    # 原始数据处理
    ###################################

    ####################
    # 对AAPR 这个数据集的处理
    ####################

    # 原始数据展示
    def show_json_data(self):
        for i in range(4):
            path = self.aapr_root + 'data{}'.format(i+1)
            # 从文件中读取一个json文件，得到一个字典（dict）变量。
            with open(path, 'r') as fp:
                data = json.load(fp)
            print(len(data))
            # 字典的遍历方式。
            for paper_id, info in data.items():
                for key, value in info.items():
                    print(key)
                break

    # 提取AAPR这个数据集中的摘要（abs）和标签（label）
    def extract_abs_label(self):
        abs_list = []
        category_list = []
        category_dict = {}
        venue_list = []
        venue_dict = {}
        label_list = []

        count = 0
        error_count = 0
        for i in range(4):
            # 总共有4个数据，为了节约演示空间，其中有3个数据是是空的。
            path = self.aapr_root + 'data{}'.format(i+1)
            # 读取数据。
            with open(path, 'r') as fp:
                data = json.load(fp)
                # 遍历数据。
                for paper_id, info in data.items():
                    abs = info['abstract'].strip()
                    category = info['category'].strip()
                    venue = info['venue'].strip()
                    if abs and category and venue:
                        # 将abs存入列表（list）中。
                        abs_list.append(abs)
                        category_list.append(category)
                        if category not in category_dict:
                            category_dict[category] = 1
                        else:
                            category_dict[category] += 1

                        venue_list.append(venue)
                        if venue not in venue_dict:
                            venue_dict[venue] = 1
                        else:
                            venue_dict[venue] += 1

                        # 将label存入列表（list）中。
                        if venue in {'CoRR', 'No'}:
                            label_list.append('0')
                        else:
                            label_list.append('1')

                    else:
                        print("Error abs: {}".format(abs))
                        print("Error label: {}".format(category))
                        print("Error venue: {}".format(venue))
                        error_count += 1
                    count += 1

        top_num = 5
        print("Print top {} abs:".format(top_num))
        for abs in abs_list[:top_num]:
            print(abs)

        print("Print top {} category:".format(top_num))
        for category in category_list[:top_num]:
            print(category)

        print("Print top {} venue:".format(top_num))
        for venue in venue_list[:top_num]:
            print(venue)

        print("category_dict:\n", category_dict)
        print("venue_dict:\n", venue_dict)

        print("There are {} papers.".format(count))
        print("There are {} error abs or labels.".format(error_count))
        return abs_list, label_list

    # 将提取的abs和label分别保存为data.input和data.output_data。
    # data.input_data, data.output_data
    def save_abs_label(self):
        save_path = self.data_root + 'aapr/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        abs_list, label_list = self.extract_abs_label()
        input_path = save_path + 'data.input_data'
        output_path = save_path + 'data.output_data'
        self.save_pair(data_input=abs_list, data_output=label_list, input_path=input_path, output_path=output_path)
        print("There are {} 1 labels.".format(sum(list(map(int, label_list)))/len(label_list)))

    ###################################
    # 通用模块
    ###################################

    # 给定一个列表（list）和一个地址，将该列表中的str数据写在这个地址对应的文件中。
    def save_single(self, data, path):
        count = 0
        # 用这种方法在文件中写入数据时，只能写入str类型的数据。
        with open(path, 'w') as fw:
            for line in data:
                fw.write(line.lower() + '\n')
                count += 1
        print("Done for saving {} lines to {}.".format(count, path))

    # 同时调用save_single。为了后面同时保存input数据和output数据。
    def save_pair(self, data_input, data_output, input_path, output_path):
        self.save_single(data_input, input_path)
        self.save_single(data_output, output_path)

    # 将上一步得到的data.input_data/output分割为三个数据集，保存为train.input_data/output_data，val.input_data/output_data，test.input_data/output_data
    # Dataset: Training set, Validation/Development set, Test set.
    def split_data(self, data_name='aapr', split_rate=0.7):
        # 将数据从地址中读取出来，得到一个列表（list）。
        with open(self.data_root + '{}/data.input_data'.format(data_name), 'r') as fp:
            data_input = list(map(lambda x: x.strip(), fp.readlines()))
            print("Successfully load input_data data from {}.".format(self.data_root + '{}/data.input_data'.format(data_name)))

        # 将数据从地址中读取出来，得到一个列表（list）。
        with open(self.data_root + '{}/data.output_data'.format(data_name), 'r') as fp:
            data_output = list(map(lambda x: x.strip(), fp.readlines()))
            print("Successfully load output_data data from {}.".format(self.data_root + '{}/data.output_data'.format(data_name)))

        # 对数据进行随机化（shuffle）。
        data = list(zip(data_input, data_output))
        random.shuffle(data)
        data_input, data_output = zip(*data)

        # 获取数据的规模，得到一个数，如30000或者50000。
        data_size = len(data_output)

        # 按照split_rate从原始数据中切出训练数据集（training set）。
        train_input = data_input[:int(data_size*split_rate)]
        train_output = data_output[:int(data_size*split_rate)]

        # 按照split_rate从原始数据中切出验证数据集（validation set）。
        val_input = data_input[int(data_size*split_rate): int(data_size*(split_rate+(1-split_rate)/2))]
        val_output = data_output[int(data_size*split_rate): int(data_size*(split_rate+(1-split_rate)/2))]

        # 按照split_rate从原始数据中切出测试数据集（testing set）。
        test_input = data_input[int(data_size*(split_rate+(1-split_rate)/2)):]
        test_output = data_output[int(data_size*(split_rate+(1-split_rate)/2)):]

        # 将切出的数据保存。
        data_folder = self.data_root + '{}/'.format(data_name)

        # 保存训练数据。
        train_input_path = data_folder + 'train.input_data'
        train_output_path = data_folder + 'train.output_data'
        self.save_pair(data_input=train_input, data_output=train_output,
                       input_path=train_input_path, output_path=train_output_path)
        print("There are {} 1 labels.".format(sum(list(map(int, train_output)))/len(train_output)))

        # 保存验证数据。
        val_input_path = data_folder + 'val.input_data'
        val_output_path = data_folder + 'val.output_data'
        self.save_pair(data_input=val_input, data_output=val_output,
                       input_path=val_input_path, output_path=val_output_path)
        print("There are {} 1 labels.".format(sum(list(map(int, val_output))) / len(val_output)))

        # 保存测试数据。
        test_input_path = data_folder + '/test.input_data'
        test_output_path = data_folder + '/test.output_data'
        self.save_pair(data_input=test_input, data_output=test_output,
                       input_path=test_input_path, output_path=test_output_path)
        print("There are {} 1 labels.".format(sum(list(map(int, test_output))) / len(test_output)))

    # 对于神经网络模型来说，需要构建词典（vocabulary）
    def get_vocab(self, data_name='aapr', cover_rate=1, mincount=1):
        data_folder = self.data_root + '{}/'.format(data_name)

        # 这个字典只能从训练数据上构建。
        # 因为验证集和测试集是在模拟未来可能碰到的情况，在模型学习时，不应该看到。
        train_input_path = data_folder + 'train.input_data'

        # 统计语料库中出现的词及词频。
        word_count_dict = {}
        # 统计语料库中总共出现了多少词。
        total_word_count = 0

        with open(train_input_path, 'r') as fp:
            for line in fp.readlines():
                for word in line.strip().split():
                    total_word_count += 1
                    if word not in word_count_dict:
                        word_count_dict[word] = 1
                    else:
                        word_count_dict[word] += 1

        # 对词按照词频进行排序。
        sorted_word_count_dict = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
        print("There are {} words originally.".format(len(sorted_word_count_dict)))

        # 按照词频顺序，对每个词赋予一个连续的id。
        # 其实这里的顺序本身并没有意义，只是为了顺便按照最低词频或者覆盖率对词典进行过滤。

        #
        word_dict = {'PAD': 0, 'UNK': 1, 'SOS': 2, 'EOS': 3}
        tmp_word_count = 0
        for word, count in sorted_word_count_dict:
            tmp_word_count += count
            # cover rate是指当前的词表可以在语料库中覆盖多大的词使用。
            current_rate = tmp_word_count / total_word_count
            # 只保留词频高于mincount和cover rate小于设定阈值的词。
            if count > mincount and current_rate < cover_rate:
                word_dict[word] = len(word_dict)
        print("There are {} words finally.".format(len(word_dict)))

        # 将词典保存。
        exp_data_folder = self.exp_root + '{}/'.format(data_name)
        if not os.path.exists(exp_data_folder):
            os.mkdir(exp_data_folder)

        word_dict_path = exp_data_folder + 'vocab.cover{}.min{}.json'.format(cover_rate, mincount)
        with open(word_dict_path, 'w') as fw:
            json.dump(word_dict, fw)
        print("Successfully save sent dict to {}.".format(word_dict_path))


# 记住哦，main是一个python脚本的入口。
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    data_processor = DataProcessor()
    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    # 在终端中执行：python data_processor.py --phase show_json_data
    elif args.phase == 'show_json_data':
        data_processor.show_json_data()
    # 在终端中执行：python data_processor.py --phase extract_abs_label
    elif args.phase == 'extract_abs_label':
        data_processor.extract_abs_label()
    # 在终端中执行：python data_processor.py --phase save_abs_label
    elif args.phase == 'save_abs_label':
        data_processor.save_abs_label()
    # 在终端中执行：python data_processor.py --phase split_data
    elif args.phase == 'split_data':
        data_processor.split_data(data_name='aapr', split_rate=0.7)
    # 在终端中执行：python data_processor.py --phase get_vocab
    elif args.phase == 'get_vocab':
        data_processor.get_vocab(data_name='aapr', cover_rate=1, mincount=1)
    else:
        print("There is no {} function. Please check your command.".format(args.phase))

    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done data_processor!')
