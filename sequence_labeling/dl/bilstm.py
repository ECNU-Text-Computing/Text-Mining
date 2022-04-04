#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Base_Model
======
A class for something.
"""
import sys
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.data_loader import DataLoader
from sequence_labeling.data_processor import DataProcessor
from sequence_labeling.utils.evaluate import Evaluator

torch.manual_seed(1)


class BiLSTM(nn.Module):

    def __init__(self, **config):
        super(BiLSTM, self).__init__()
        self.config = config
        self.data_root = config['data_root']
        self.model_save_path = config['model_save_name']

        self.pad_token = config['pad_token']

        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.bidirectional = config['bidirectional']
        self.num_layers = config['num_layers=']
        self.tagset_size = config['tag_size']
        self.num_epochs = config['num_epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout_rate']

        self.criterion_dict = {
            'NLLLoss': torch.nn.NLLLoss,
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss
        }
        self.criterion_name = config['criterion_name']
        if self.criterion_name not in self.criterion_dict:
            raise ValueError("There is no criterion_name: {}.".format(self.criterion_name))

        self.optimizer_dict = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam
        }
        self.optimizer_name = config['optimizer_name']
        if self.optimizer_name not in self.optimizer_dict:
            raise ValueError("There is no optimizer_name: {}.".format(self.optimizer_name))

        vocab = DataProcessor(**self.config).load_vocab()
        vocab_size = len(vocab)
        padding_idx = vocab[self.pad_token]
        self.word_embeddings = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=padding_idx)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=self.dropout)

        # The linear layer that maps from hidden state space to tag space
        self.hidden_to_tag = nn.Linear(self.hidden_dim*2, self.tagset_size)

        self.hidden = self.init_hidden(self.batch_size)

    def init_hidden(self, batch_size):
        # 各个维度的含义是 (num_layers*num_directions, batch_size, hidden_dim)
        # 避免input数据条数<self.batch_size。h0初始化时batch_size的值必须等于batch的实际大小
        return (torch.randn(self.num_layers * 2, batch_size, self.hidden_dim),
                torch.randn(self.num_layers * 2, batch_size, self.hidden_dim))

    def forward(self, X, X_lengths):
        batch_size, seq_len = X.size()
        X = self.word_embeddings(X)
        X = rnn_utils.pack_padded_sequence(X, X_lengths, batch_first=True)
        X, _ = self.lstm(X, self.hidden)
        X, _ = rnn_utils.pad_packed_sequence(X, batch_first=True)
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        X = self.hidden_to_tag(X)
        tag_scores = F.log_softmax(X, dim=1)
        # print('tag_scores:{}'.format(tag_scores.shape))
        # tag_scores = X.view(batch_size, seq_len, self.tagset_size)
        # print('tag_scores_view:{}'.format(tag_scores.shape))
        return tag_scores

    def run_model(self, model, run_mode, data_path):
        print('Running Bi-LSTM model...')
        loss_function = self.criterion_dict[self.criterion_name]()
        optimizer = self.optimizer_dict[self.optimizer_name](model.parameters(), lr=self.learning_rate)

        if run_mode == 'train':
            model.train()
            for epoch in range(self.num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
                # for sentenceList, tagList in DataLoader().data_generator(batch_size=self.batch_size, op_mode=op_mode):
                for x, x_len, y, y_len in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                                   run_mode=run_mode):
                    batch_x = torch.tensor(x).long()
                    model.zero_grad()
                    model.hidden = model.init_hidden(len(x))
                    tag_scores = model(batch_x, x_len)
                    batch_y = torch.tensor(y).long()
                    batch_y = batch_y.view(-1)
                    # print('batch_y:{}'.format(batch_y.shape))
                    loss = loss_function(tag_scores, batch_y)
                    loss.backward()
                    optimizer.step()
                print("Done Epoch{}!".format(epoch))

            model_save_path = self.data_root + self.model_save_path
            torch.save(model, '{}'.format(model_save_path))
        elif run_mode == 'eval' or 'test':
            model.eval()
            with torch.no_grad():
                for x, x_len, y, y_len in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                                   run_mode=run_mode):
                    batch_x = torch.tensor(x).long()
                    model.hidden = model.init_hidden(len(x))
                    tag_scores = model(batch_x, x_len)
                    # print("After Train:", scores)
                    # print('标注结果转换为Tag索引序列：', torch.max(scores, dim=1))
                    y_predict = list(torch.max(tag_scores, dim=1)[1].numpy())
                    y_predict = self.index_to_tag(y_predict)
                    print(y_predict)

                    # print正确的标注结果
                    y_true = y.flatten()
                    y_true = self.index_to_tag(y_true)
                    print(y_true)

                    # 输出评价结果
                    print(Evaluator().classifyreport(y_true, y_predict))
        else:
            print("run_mode参数未赋值(train/eval/test)")

    def index_to_tag(self, y):
        tag_index_dict = DataProcessor(**self.config).load_tags()
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
