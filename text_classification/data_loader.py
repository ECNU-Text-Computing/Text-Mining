#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Data_Loader
======
A class for something.
"""

import os
import random
import sys
import argparse
import datetime

import torch

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from data_processor import DataProcessor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from joblib import load, dump
import numpy as np
# from keras.preprocessing.sequence import pad_sequences
from torch.nn.utils.rnn import pad_sequence


class DataLoader(DataProcessor):
    def __init__(self):
        super(DataLoader, self).__init__()

    # 将文本转化为机器学习模型需要的数据格式。
    # 对于机器学习模型来说，需要提取相应的特征，如向量空间模型或者主题模型。此处暂时不考虑词向量。
    def data_load(self, data_name='aapr', phase='train', feature='tf', **kwargs):
        # 获取数据地址，并将之读取成列表（list）。
        input_path = '{}{}/{}.input_data'.format(self.data_root, data_name, phase)
        output_path = '{}{}/{}.output'.format(self.data_root, data_name, phase)
        with open(input_path, 'r') as fp:
            input_data = list(map(lambda x: x.strip(), fp.readlines()))
        with open(output_path, 'r') as fp:
            output_data = list(map(lambda x: int(x.strip()), fp.readlines()))

        # 如果你的机器资源不够，一次跑不完数据，为了验证整个流程（pipeline）是对的，可以采用小一点的数据规模。
        # 比方说，这里设置了1000，你可以可以设置100或者200等。
        # 如果你可以跑全量数据，那么你可以注释掉这行代码。
        # 记得有一些可用可不用的代码，可以注释掉而不要直接删掉。免得以后用的时候还要重新写。
        # src = src[:1000]
        # out = out[:1000]

        # 将此处提取的特征（tf、tf-idf或lda）保存下来。
        save_folder = self.exp_root + data_name
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if not os.path.exists(save_folder + '/ml_feature/'):
            os.mkdir(save_folder + '/ml_feature/')
        save_path = save_folder + '/ml_feature/{}'.format(feature)

        # 首先是输入数据：

        # 如果是训练模型，则需要完整的执行下面的代码，用来提取数据中的特征。
        # if phase == 'train' and not os.path.exists(save_path):
        if phase == 'train':
            if feature == 'tf':
                # 其中包括了字典构建和词频统计。
                feature_extractor = CountVectorizer().fit(input_data)
            elif feature == 'tfidf':
                # 其中包括了字典构建和词频、文档频率的统计和tf-idf的计算。
                feature_extractor = TfidfVectorizer().fit(input_data)
            elif feature == 'lda':
                # 构建LDA的词典。
                dictionary = Dictionary([text.strip().split() for text in input_data])
                # 保存此词典。
                dictionary_save_path = save_path + '.dict'
                if not os.path.exists(dictionary_save_path):
                    dump(dictionary, dictionary_save_path)
                    print("Successfully save dict to {}.".format(dictionary_save_path))
                # 利用此词典将原来的字符序列转化为id序列。
                corpus = [dictionary.doc2bow(text.strip().split()) for text in input_data]
                # 主题个数的确认。
                num_topics = 20
                # if 'num_topics' in kwargs:
                #     num_topics = kwargs['num_topics']
                # 主题模型的训练。
                feature_extractor = LdaModel(corpus, num_topics=num_topics)
            else:
                raise RuntimeError("Please confirm which feature you need.")
            if not os.path.exists(save_path):
                dump(feature_extractor, save_path)
                print("Successfully save features to {}.".format(save_path))
        # 如果不是训练，则可以直接加载上述已经保存好的"模型"。
        # 注意，上面tf和tf-idf不算是模型，只是一些规则；而lda是一个模型。
        # 具体的差异可以自行深入学习。
        else:
            feature_extractor = load(save_path)

        # 如果用lda的方式提取特征，则可以用以下方法，将每个样本表示为一个主题的分布（向量）。
        # 此时，所有样本的向量可以构成一个矩阵：文档-主题。
        if feature == 'lda':
            dictionary = load(save_path + '.dict')
            x = [feature_extractor.get_document_topics(dictionary.doc2bow(text.strip().split()), minimum_probability=0)
                 for text in input_data]
            x = [[prob for (topic, prob) in line] for line in x]
        # tf或者tf-idf的方法提取特征，则可以用以下方法，将每个样本表示为一个词表长度大小的向量。
        # 此时，所有样本的向量可以构成一个矩阵：文档-词项。
        else:
            x = feature_extractor.transform(input_data)

        # 然后是输出数据：

        # 输出数据应该是label的id。
        # 如果是二分类，应该是[0, 1, 1, ..., 0];
        # 如果是四分类，则应该是[0, 1, 2, 3, 1, ..., 1]。
        # 其他多分类，以此类推。
        y = output_data

        return x, y

    # 将文本转化为深度学习模型需要的数据格式。
    def data_generator(self, input_path, output_path, feature,
                       word_dict=None, batch_size=64, pad_size=16, shuffle=True):

        print("Load input_data data from {}.".format(input_path))
        print("Load output_data from {}.".format(output_path))

        # 读取输入数据。
        # type(src) = list
        with open(input_path, 'r') as fp:
            input_data = fp.readlines()

        # 读取输出数据。
        # type(out) = list
        with open(output_path, 'r') as fp:
            output_data = fp.readlines()

        # 如果你的机器资源不够，一次跑不完数据，为了验证整个流程（pipeline）是对的，可以采用小一点的数据规模。
        # 比方说，这里设置了1000，你可以可以设置100或者200等。
        # 如果你可以跑全量数据，那么你可以注释掉这行代码。
        # 记得有一些可用可不用的代码，可以注释掉而不要直接删掉。免得以后用的时候还要重新写。
        # src = src[:1000]
        # out = out[:1000]

        # 是否将数据随机打乱？
        if shuffle:
            data = list(zip(input_data, output_data))
            random.shuffle(data)
            input_data, output_data = zip(*data)

        # 按照batch将数据输出给深度学习模型。
        # 注意，因为下面使用了yield而非return，所以这个函数是一个生成器。具体的使用见深度学习Base_Model的train部分。
        if feature == 'hierarchical':
            for i in range(0, len(input_data), batch_size):
                new_batch = []
                # 读取输入数据的下一个batch。
                for doc in input_data[i: i + batch_size]:
                    new_doc = []
                    # 将该文档的句子拆分开
                    for sent in doc.strip().split('.'):
                        new_sent = []
                        # 将该句子的单词拆分开，并将字符的列表转化为id的列表。
                        for word in sent.strip().split():
                            new_sent.append(int(word_dict[word]) if word in word_dict else int(word_dict['UNK']))
                        # 对于seq_len的padding操作
                        while True:
                            if len(new_sent) > pad_size:
                                new_doc.append(torch.Tensor(new_sent[:pad_size]))
                                new_sent = new_sent[pad_size:-1]
                            elif len(new_sent) < pad_size:
                                lst = [0] * int(pad_size - len(new_sent))
                                new_sent.extend(lst)
                                new_doc.append(torch.Tensor(new_sent))
                                break
                            else:
                                new_doc.append(torch.Tensor(new_sent))
                                break
                    # 将转化后的id列表append到batch_input中。
                    # 因为要使用toch的pad函数，所以此处就要把每个id列表转化为torch的Tensor格式。
                    new_batch.append(torch.stack(new_doc))
                batch_x = pad_sequence(new_batch, batch_first=True).detach().numpy()
                # 提取输出数据的下一个batch。
                batch_output = [int(label.strip()) for label in output_data[i: i + batch_size]]
                batch_y = np.array(batch_output)

                yield batch_x, batch_y

        else:
            for i in range(0, len(output_data), batch_size):
                # 每次循环开始时，都先清空batch_input。
                batch_input = []
                # 读取输入数据的下一个batch。
                for line in input_data[i: i + batch_size]:
                    new_line = []
                    # 将字符的列表转化为id的列表。
                    for word in line.strip().split():
                        # 如果有数据不在字典中，则使用UNK的id。
                        new_line.append(int(word_dict[word]) if word in word_dict else int(word_dict['UNK']))
                    # 将转化后的id列表append到batch_input中。
                    # 因为要使用toch的pad函数，所以此处就要把每个id列表转化为torch的Tensor格式。
                    batch_input.append(torch.Tensor(new_line))
                # batch_x = pad_sequences(new_batch, maxlen=None)
                # 将不等长的数据进行pad操作。
                # [[1, 2], [1, 2, 3]] ==> [[1, 2, 0], [1, 2, 3]]
                batch_x = pad_sequence(batch_input, batch_first=True).detach().numpy()
                # 提取输出数据的下一个batch。
                batch_output = [int(label.strip()) for label in output_data[i: i + batch_size]]
                batch_y = np.array(batch_output)

                yield batch_x, batch_y


# 记住哦，main是一个python脚本的入口。
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

    print('Done Data_Loader!')
