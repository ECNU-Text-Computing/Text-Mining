#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
BiLSTM_CRF
======
A class for something.
"""

import os
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(1)

import numpy as np

from sequence_labeling.dl.base_model import BaseModel
from sequence_labeling.data_loader import Data_Loader
from sequence_labeling.data_processor import DataProcessor
from sequence_labeling.utils.evaluate import Evaluator
from .crf import CRF


class BiLSTM_CRF(BaseModel):

    def __init__(self, data_root, model_save_path, embedding_dim, hidden_dim, tag_size, vocab_size,
                 tag_to_ix,
                 num_epochs, batch_size, learning_rate, criterion_name, optimizer_name, **kwargs): #  **kwargs可变关键词参数，列表

        super(BiLSTM_CRF, self).__init__(data_root, model_save_path, embedding_dim, hidden_dim, tag_size, vocab_size,
                 num_epochs, batch_size, learning_rate, criterion_name, optimizer_name, **kwargs)

        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(self.tag_to_ix)

        self.hidden = self.init_hidden()
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.crf = CRF(self.tagset_size, self.tag_to_ix)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))



    # 从三层网络中得到feats
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats


    # 根据viterbi解码得到预测
    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self.crf._viterbi_decode(lstm_feats)
        return lstm_feats, torch.tensor(tag_seq)


    def run_Model(self, model, op_mode):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

        if op_mode == 'train':
            model.train()
            for epoch in range(self.num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
                for sentenceList, tagList in Data_Loader().data_generator(batch_size=self.batch_size, op_mode=op_mode):
                    for i in range(len(sentenceList)):
                        batch_x = torch.LongTensor(sentenceList[i])
                        batch_y = torch.LongTensor(tagList[i])
                        model.zero_grad()
                        tag_scores = model(batch_x)[0]
                        loss =self.crf.neg_log_likelihood(tag_scores, batch_y)
                        loss.backward()
                        optimizer.step()

            model_save_path = self.data_root + self.model_save_path
            torch.save(model, '{}'.format(model_save_path))
        elif op_mode == 'eval' or 'test':
            model.eval()
        else:
            print("op_mode参数未赋值(train/eval/test)")

        with torch.no_grad():
            for sentenceList, tagList in Data_Loader().data_generator(op_mode=op_mode):
                batch_x = torch.LongTensor(sentenceList[0])
                print(batch_x)

                # print("After Train:", scores)
                # print('标注结果转换为Tag索引序列：', torch.max(scores, dim=1))
                y_predict =list(model(batch_x)[1].numpy())

                # 将索引序列转换为Tag序列
                y_pred = self.index_to_tag(y_predict)
                y_true = self.index_to_tag(tagList[0])

                # 输出评价结果
                print(y_pred)
                print(y_true)
                print(Evaluator().acc(y_true, y_pred))


    def index_to_tag(self, y):
        tag_index_dict = DataProcessor().load_tags()
        index_tag_dict = dict(zip(tag_index_dict.values(), tag_index_dict.keys()))
        y_tagseq = []
        for i in range(len(y)):
            y_tagseq.append(index_tag_dict[y[i]])
        return y_tagseq

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

    print('Done Base_Model!')
