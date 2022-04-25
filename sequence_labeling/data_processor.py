#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
DataProcessor
======
A class for something.
"""

import datetime
import json
import random
import os


class DataProcessor(object):
    def __init__(self, **config):
        super(DataProcessor, self).__init__()
        # print('Data_Processor Class Init...')
        self.data_root = config['data_root']
        self.inputdata_file_name = config['inputdata_file_name']
        self.outputdata_file_name = config['outputdata_file_name']
        self.dict_file_name = config['dict_file_name']
        self.tags_file_name = config['tags_file_name']

        self.n_folds = config['n_folds']
        self.split_rate = config['split_rate']
        self.cover_rate = config['cover_rate']
        self.min_count = config['min_count']

    # 生成训练集、验证集和测试集
    def generate_datasets(self):
        if self.n_folds == 0:
            train_input, train_output, eval_input, eval_output, test_input, test_output = self.split_data()
            train_inputdata_path = '{}{}.train'.format(self.data_root, self.inputdata_file_name)
            train_outputdata_path = '{}{}.train'.format(self.data_root, self.outputdata_file_name)
            eval_inputdata_path = '{}{}.eval'.format(self.data_root, self.inputdata_file_name)
            eval_outputdata_path = '{}{}.eval'.format(self.data_root, self.outputdata_file_name)
            test_inputdata_path = '{}{}.test'.format(self.data_root, self.inputdata_file_name)
            test_outputdata_path = '{}{}.test'.format(self.data_root, self.outputdata_file_name)

            self.save(train_input, train_inputdata_path)
            self.save(train_output, train_outputdata_path)
            self.save(eval_input, eval_inputdata_path)
            self.save(eval_output, eval_outputdata_path)
            self.save(test_input, test_inputdata_path)
            self.save(test_output, test_outputdata_path)
        else:
            for n in range(self.n_folds):
                n_fold_dataset_path = '{}{}/'.format((self.data_root, n))
                if not os.path.exists(n_fold_dataset_path):
                    os.mkdir(n_fold_dataset_path)

                train_input, train_output, eval_input, eval_output, test_input, test_output = self.split_data()
                train_inputdata_path = '{}{}.train'.format(n_fold_dataset_path, self.inputdata_file_name)
                train_outputdata_path = '{}{}.train'.format(n_fold_dataset_path, self.outputdata_file_name)
                eval_inputdata_path = '{}{}.eval'.format(n_fold_dataset_path, self.inputdata_file_name)
                eval_outputdata_path = '{}{}.eval'.format(n_fold_dataset_path, self.outputdata_file_name)
                test_inputdata_path = '{}{}.test'.format(n_fold_dataset_path, self.inputdata_file_name)
                test_outputdata_path = '{}{}.test'.format(n_fold_dataset_path, self.outputdata_file_name)

                self.save(train_input, train_inputdata_path)
                self.save(train_output, train_outputdata_path)
                self.save(eval_input, eval_inputdata_path)
                self.save(eval_output, eval_outputdata_path)
                self.save(test_input, test_inputdata_path)
                self.save(test_output, test_outputdata_path)

    # 将data.input和data.output划分为训练集、验证集和测试集
    def split_data(self):
        with open(self.data_root + self.inputdata_file_name, 'r', encoding='utf-8') as fp:
            data_input = list(map(lambda x: x.strip(), fp.readlines()))
        with open(self.data_root + self.outputdata_file_name, 'r', encoding='utf-8') as fp:
            data_output = list(map(lambda x: x.strip(), fp.readlines()))

        data = list(zip(data_input, data_output))
        random.shuffle(data)
        data_input, data_output = zip(*data)

        data_size = len(data_output)
        train_data_input = data_input[:int(data_size * self.split_rate)]
        train_data_output = data_output[:int(data_size * self.split_rate)]
        val_data_input = data_input[
                         int(data_size * self.split_rate): int(
                             data_size * (self.split_rate + (1 - self.split_rate) / 2))]
        val_data_output = data_output[
                          int(data_size * self.split_rate): int(
                              data_size * (self.split_rate + (1 - self.split_rate) / 2))]
        test_data_input = data_input[int(data_size * (self.split_rate + (1 - self.split_rate) / 2)):]
        test_data_output = data_output[int(data_size * (self.split_rate + (1 - self.split_rate) / 2)):]

        return train_data_input, train_data_output, val_data_input, val_data_output, test_data_input, test_data_output

    # 将数据存入文件
    def save(self, data, filepath):
        count = 0
        with open(filepath, 'w', encoding='utf-8') as fw:
            for line in data:
                fw.write(line + '\n')
                count += 1
        # print("Done for saving {} lines to {}.".format(count, filepath))

    # 依据训练数据构建词索引词典(word:id)
    def get_vocab(self):
        data_input_path = self.data_root + self.inputdata_file_name

        word_count_dict = {}
        total_word_count = 0
        with open(data_input_path, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                for word in line.strip().split():
                    total_word_count += 1
                    if word not in word_count_dict:
                        word_count_dict[word] = 1
                    else:
                        word_count_dict[word] += 1
        sorted_word_count_dict = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
        print("There are {} words originally.".format(len(sorted_word_count_dict)))

        word_dict = {'<PAD>': 0, 'UNK': 1, 'SOS': 2, 'EOS': 3}
        tmp_word_count = 0
        for word, count in sorted_word_count_dict:
            print("{}:{}".format(word, count))
            tmp_word_count += count
            current_rate = tmp_word_count / total_word_count
            if count > self.min_count and current_rate <= self.cover_rate:
                word_dict[word] = len(word_dict)
        print("There are {} words finally.".format(len(word_dict)))

        word_dict_path = self.data_root + self.dict_file_name
        with open(word_dict_path, 'w', encoding='utf-8') as fw:
            json.dump(word_dict, fw)
        print("Successfully save word dict to {}.".format(word_dict_path))

    # 读取vocab.json的内容，返回值为{}类型。
    def load_vocab(self):
        word_dict_path = self.data_root + self.dict_file_name
        with open(word_dict_path, 'r', encoding='utf-8') as fp:
            word_dict = json.load(fp)
            # print("Load word dict from {}.".format(word_dict_path))
            return word_dict

    # 读取tags.json的内容，返回值为{}类型。
    def load_tags(self):
        tagfile_path = self.data_root + self.tags_file_name
        with open(tagfile_path, 'r', encoding='utf-8') as tfp:
            tags_dict = json.load(tfp)
            # print("Load Tags dict from {}.".format(tagfile_path))
            return tags_dict


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    config_path = './config/cmed/dl/cmed.dl.base_model.norm.json'
    if not os.path.exists(config_path):
        raise RuntimeError("There is no {} config.".format(config_path))
    config = json.load(open(config_path, 'r', encoding='utf-8'))
    print('config: ', config)

    data_processor = DataProcessor(**config)

    # 生成词典vocab.json
    data_processor.get_vocab()
    print(data_processor.load_vocab())

    # 将data.input和data.output划分成训练集、验证集和测试集
    data_processor.generate_datasets()

    # print(data_processor.load_tags())

    end_time = datetime.datetime.now()
    print('dataProcessor takes {} seconds.'.format((end_time - start_time).seconds))
