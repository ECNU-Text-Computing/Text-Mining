#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
DataLoader
======
A class for something.
"""

import os
import sys
import random
import argparse
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from joblib import load, dump
import numpy as np
import json
from keras.preprocessing.sequence import pad_sequences
# from keras.utils.np_utils import to_categorical
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from text_classification.data_processor import DataProcessor


class DataLoader(DataProcessor):
    def __init__(self):
        super(DataLoader, self).__init__()

    # tf-idf, lda
    def data_load(self, data_name='aapr', phase='train', fold=0, feature='tf', clean=0, clear=0, *args, **kwargs):
        if clean:
            mode = '_'.join(['clean'])
            input_path = '{}{}/{}/{}_{}_{}.input'.format(self.data_root, data_name, fold, phase, mode, fold)
        else:
            input_path = '{}{}/{}/{}_{}.input'.format(self.data_root, data_name, fold, phase, fold)
        output_path = '{}{}/{}/{}_{}.output'.format(self.data_root, data_name, fold, phase, fold)
        with open(input_path, 'r') as fp:
            input_data = list(map(lambda x: x.strip(), fp.readlines()))
        with open(output_path, 'r') as fp:
            output_data = list(map(lambda x: int(x.strip()), fp.readlines()))

        save_folder = self.exp_root + data_name
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if not os.path.exists(save_folder + '/ml_feature/'):
            os.mkdir(save_folder + '/ml_feature/')
        save_path = save_folder + '/ml_feature/{}.{}'.format(feature, fold)

        if phase == 'train' and (not os.path.exists(save_path) or clear):
            if feature == 'tf':
                feature_extractor = CountVectorizer().fit(input_data)
            elif feature == 'tfidf':
                feature_extractor = TfidfVectorizer().fit(input_data)
            elif feature == 'lda':
                dictionary = Dictionary([text.strip().split() for text in input_data])
                dictionary_save_path = save_path + '.dict'
                if not os.path.exists(dictionary_save_path) or clear:
                    dump(dictionary, dictionary_save_path)
                    print("Successfully save dict to {}.".format(dictionary_save_path))
                corpus = [dictionary.doc2bow(text.strip().split()) for text in input_data]
                num_topics = 20
                if 'num_topics' in kwargs:
                    num_topics = kwargs['num_topics']
                feature_extractor = LdaModel(corpus, num_topics=num_topics)
            else:
                raise RuntimeError("Please confirm which feature you need.")
            if not os.path.exists(save_path) or clear:
                dump(feature_extractor, save_path)
                print("Successfully save features to {}.".format(save_path))
        else:
            feature_extractor = load(save_path)

        if feature == 'lda':
            dictionary = load(save_path+'.dict')
            x = [feature_extractor.get_document_topics(dictionary.doc2bow(text.strip().split()), minimum_probability=0)
                 for text in input_data]
            x = [[prob for (topic, prob) in line] for line in x]
        else:
            x = feature_extractor.transform(input_data)
        y = output_data

        return x, y

    def data_generator(self, input_path, output_path,
                       word_dict=None, batch_size=64, shuffle=True):

        print("Load input data from {}.".format(input_path))
        print("Load output data from {}.".format(output_path))
        # type(input_data) = list
        with open(input_path, 'r') as fp:
            input_data = fp.readlines()

        # type(output_data) = list
        with open(output_path, 'r') as fp:
            output_data = fp.readlines()

        if shuffle:
            data = list(zip(input_data, output_data))
            random.shuffle(data)
            input_data, output_data = zip(*data)

        # len(output_data) = 20000
        # batch_size = 64
        # [0, 64, 128,  ... 20000]
        # input_data[i:i+batch_size]会发生数组越界吗
        for i in range(0, len(output_data), batch_size):
            batch_input = []
            # [0, 64]
            # [64, 128]
            # input_data[i: ]
            for line in input_data[i: i+batch_size]:
                new_line = []
                # '我 爱 中国'
                # ['我', '爱', '中国']
                # [4, 5, 6]
                for word in line.strip().split():
                    new_line.append(int(word_dict[word]) if word in word_dict else int(word_dict['UNK']))
                batch_input.append(new_line)
            # [[4, 5, 6], [4, 5, 7], [5, 5, 6, 8, 9, 10], ....]
            # [[4, 5, 6, 0, 0, 0, ], [4, 5, 7, 0, 0], [5, 5, 6, 8, 9], ....]
            # type(batch_x) = numpy.ndarray
            # non keras
            batch_x = pad_sequences(batch_input, maxlen=None)
            batch_output = [int(label.strip()) for label in output_data[i: i+batch_size]]
            batch_y = np.array(batch_output)
            # batch_x: [batch_size, seq_len]
            # batch_y: [batch_size, 1]
            # yield: generator
            yield batch_x, batch_y


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done data_loader!')
