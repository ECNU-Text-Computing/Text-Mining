#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
DataProcessor
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
        self.data_root = './datasets/'
        self.original_root = self.data_root + 'original/'
        self.aapr_root = self.original_root + 'aapr_dataset/'

        self.exp_root = './exp/'

    ##############################
    # AAPR
    ##############################
    def show_json_data(self):
        for i in range(4):
            path = self.aapr_root + 'data{}'.format(i+1)
            with open(path, 'r') as fp:
                data = json.load(fp)
            print(len(data))
            for paper_id, info in data.items():
                for key, value in info.items():
                    print(key)
                break

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
            path = self.aapr_root + 'data{}'.format(i+1)
            with open(path, 'r') as fp:
                data = json.load(fp)
                for paper_id, info in data.items():
                    abs = info['abstract'].strip()
                    category = info['category'].strip()
                    venue = info['venue'].strip()
                    if abs and category and venue:
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

    def save_single(self, data, path, clean=0):
        count = 0
        with open(path, 'w') as fw:
            for line in data:
                if clean:
                    line = self.clean_line(line)
                fw.write(line + '\n')
                count += 1
        print("Done for saving {} lines to {}.".format(count, path))

    def save_pair(self, data_input, data_output, input_path, output_path, clean=0):
        self.save_single(data_input, input_path, clean=clean)
        self.save_single(data_output, output_path, clean=clean)

    # data.input, data.output
    def save_abs_label(self):
        save_path = self.data_root + 'aapr/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        abs_list, label_list = self.extract_abs_label()
        input_path = save_path + 'data.input'
        output_path = save_path + 'data.output'
        self.save_pair(data_input=abs_list, data_output=label_list, input_path=input_path, output_path=output_path)
        print("There are {} 1 labels.".format(sum(list(map(int, label_list)))/len(label_list)))

    ################################################################################

    def clean_line(self, line):
        new_line = nltk.word_tokenize(line.lower())
        return ' '.join(new_line)

    def split_data(self, data_name='aapr', fold=10, split_rate=0.7, clean=0, *args, **kwargs):
        with open(self.data_root + '{}/data.input'.format(data_name), 'r') as fp:
            data_input = list(map(lambda x: x.strip(), fp.readlines()))
            print("Successfully load input data from {}.".format(self.data_root + '{}/data.input'.format(data_name)))

        with open(self.data_root + '{}/data.output'.format(data_name), 'r') as fp:
            data_output = list(map(lambda x: x.strip(), fp.readlines()))
            print("Successfully load output data from {}.".format(self.data_root + '{}/data.output'.format(data_name)))

        for i in range(fold):
            print("Processing fold {}...".format(i))
            random.seed(i)
            data = list(zip(data_input, data_output))
            random.shuffle(data)
            data_input, data_output = zip(*data)

            data_size = len(data_output)
            train_input = data_input[:int(data_size*split_rate)]
            train_output = data_output[:int(data_size*split_rate)]
            val_input = data_input[int(data_size*split_rate): int(data_size*(split_rate+(1-split_rate)/2))]
            val_output = data_output[int(data_size*split_rate): int(data_size*(split_rate+(1-split_rate)/2))]
            test_input = data_input[int(data_size*(split_rate+(1-split_rate)/2)):]
            test_output = data_output[int(data_size*(split_rate+(1-split_rate)/2)):]

            data_folder = self.data_root + '{}/'.format(data_name)
            data_fold_folder = data_folder + '{}/'.format(i)
            if not os.path.exists(data_fold_folder):
                os.mkdir(data_fold_folder)

            if clean:
                mode = '_'.join(['clean'])
                train_input_path = data_fold_folder + 'train_{}_{}.input'.format(mode, i)
                train_output_path = data_fold_folder + 'train_{}_{}.output'.format(mode, i)
            else:
                train_input_path = data_fold_folder + 'train_{}.input'.format(i)
                train_output_path = data_fold_folder + 'train_{}.output'.format(i)
            self.save_pair(data_input=train_input, data_output=train_output,
                           input_path=train_input_path, output_path=train_output_path,
                           clean=clean)
            print("There are {} 1 labels.".format(sum(list(map(int, train_output)))/len(train_output)))

            if clean:
                mode = '_'.join(['clean'])
                val_input_path = data_fold_folder + 'val_{}_{}.input'.format(mode, i)
                val_output_path = data_fold_folder + 'val_{}_{}.output'.format(mode, i)
            else:
                val_input_path = data_fold_folder + 'val_{}.input'.format(i)
                val_output_path = data_fold_folder + 'val_{}.output'.format(i)
            self.save_pair(data_input=val_input, data_output=val_output,
                           input_path=val_input_path, output_path=val_output_path, clean=clean)
            print("There are {} 1 labels.".format(sum(list(map(int, val_output))) / len(val_output)))

            if clean:
                mode = '_'.join(['clean'])
                test_input_path = data_fold_folder + '/test_{}_{}.input'.format(mode, i)
                test_output_path = data_fold_folder + '/test_{}_{}.output'.format(mode, i)
            else:
                test_input_path = data_fold_folder + '/test_{}.input'.format(i)
                test_output_path = data_fold_folder + '/test_{}.output'.format(i)
            self.save_pair(data_input=test_input, data_output=test_output,
                           input_path=test_input_path, output_path=test_output_path, clean=clean)
            print("There are {} 1 labels.".format(sum(list(map(int, test_output))) / len(test_output)))

    def get_vocab(self, data_name='aapr', fold=10, clean=0, cover_rate=1, mincount=0, *args, **kwargs):
        data_folder = self.data_root + '{}/'.format(data_name)

        for i in range(fold):
            data_fold_folder = data_folder + '{}/'.format(i)
            if clean:
                mode = '_'.join(['clean'])
                train_input_path = data_fold_folder + 'train_{}_{}.input'.format(mode, i)
            else:
                train_input_path = data_fold_folder + 'train_{}.input'.format(i)

            word_count_dict = {}
            total_word_count = 0
            with open(train_input_path, 'r') as fp:
                for line in fp.readlines():
                    for word in line.strip().split():
                        total_word_count += 1
                        if word not in word_count_dict:
                            word_count_dict[word] = 1
                        else:
                            word_count_dict[word] += 1
            sorted_word_count_dict = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
            print("There are {} words originally.".format(len(sorted_word_count_dict)))
            word_dict = {'PAD': 0, 'UNK': 1, 'SOS': 2, 'EOS': 3}
            tmp_word_count = 0
            for word, count in sorted_word_count_dict:
                tmp_word_count += count
                current_rate = tmp_word_count / total_word_count
                if count > mincount and current_rate < cover_rate:
                    word_dict[word] = len(word_dict)
            print("There are {} words finally.".format(len(word_dict)))
            exp_data_folder = self.exp_root + '{}/'.format(data_name)
            if not os.path.exists(exp_data_folder):
                os.mkdir(exp_data_folder)
            exp_data_dl_folder = exp_data_folder + 'dl/'
            if not os.path.exists(exp_data_dl_folder):
                os.mkdir(exp_data_dl_folder)
            vocal_data_dl_folder = exp_data_dl_folder + 'vocab/'
            if not os.path.exists(vocal_data_dl_folder):
                os.mkdir(vocal_data_dl_folder)
            word_dict_path = vocal_data_dl_folder + 'vocab.cover{}.min{}.{}.json'.format(cover_rate, mincount, i)
            with open(word_dict_path, 'w') as fw:
                json.dump(word_dict, fw)
            print("Successfully save word dict to {}.".format(word_dict_path))


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    data_processor = DataProcessor()
    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    elif args.phase == 'show_json_data':
        data_processor.show_json_data()
    elif args.phase == 'extract_abs_label':
        data_processor.extract_abs_label()
    elif args.phase == 'save_abs_label':
        data_processor.save_abs_label()
    elif args.phase.split('+')[0] == 'split_data':
        config_name = args.phase.split('+')[1]
        data_name = config_name.strip().split('.')[0]
        model_cate = config_name.strip().split('.')[1]
        config_path = './config/{}/{}/{}.json'.format(data_name, model_cate, config_name)
        config = json.load(open(config_path, 'r'))
        data_processor.split_data(**config)
    elif args.phase.split('+')[0] == 'get_vocab':
        config_name = args.phase.split('+')[1]
        data_name = config_name.strip().split('.')[0]
        model_cate = config_name.strip().split('.')[1]
        config_path = './config/{}/{}/{}.json'.format(data_name, model_cate, config_name)
        config = json.load(open(config_path, 'r'))
        data_processor.get_vocab(**config)
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done data_processor!')
