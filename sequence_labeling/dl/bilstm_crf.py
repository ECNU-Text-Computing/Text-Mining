#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
BiLSTM_CRF
======
A class for something.
"""

import argparse
import datetime
import json
import sys

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from sequence_labeling.data_processor import DataProcessor
from sequence_labeling.data_loader import DataLoader
from sequence_labeling.utils.evaluate import Evaluator
from .crf import CRF

torch.manual_seed(1)

class BiLSTM_CRF(nn.Module):

    def __init__(self, **config):
        super(BiLSTM_CRF, self).__init__()
        self.config = config

        # 将"<START>"、"<STOP>"加入Tag字典
        self.data_root = config['data_root']
        self.model_save_path = config['model_save_name']
        self.tags_file_name = config['tags_file_name']
        tagfile_path = self.data_root + self.tags_file_name
        with open(tagfile_path, 'r') as tfp:
            self.tag_to_ix = json.load(tfp)
        self.tag_to_ix['<START>'] = len(self.tag_to_ix)
        self.tag_to_ix['<STOP>'] = len(self.tag_to_ix)
        self.tagset_size = len(self.tag_to_ix)

        self.model_name = config['model_name']
        self.bidirectional = True if config['bidirectional'] == 'True' else False
        self.embedding_dim = config['embedding_dim']
        self.pad_token = config['pad_token']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers=']
        self.learning_rate = config['learning_rate']
        self.num_epochs = config['num_epochs']

        self.crf = CRF(self.tagset_size, self.tag_to_ix)
        self.criterion_dict = {
            'Neg_Log_Likelihood': self.crf.neg_log_likelihood
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

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True, )

        self.n_directions = 2 if self.bidirectional else 1  # 双向循环，输出的hidden是正向和反向hidden的拼接，所以要 *2
        self.hidden_to_tag = nn.Linear(self.hidden_dim * self.n_directions, self.tagset_size)



    def _init_hidden(self, batch_size):  # 初始化h_0
        hidden = (torch.zeros(self.num_layers * self.n_directions, batch_size, self.hidden_dim),
                  torch.zeros(self.num_layers * self.n_directions, batch_size, self.hidden_dim))
        return hidden

    def forward(self, X, X_lengths, run_mode):
        batch_size, seq_len = X.size()
        # batch_size = 1  # 单句处理
        hidden = self._init_hidden(batch_size)
        X = self.word_embeddings(X)
        X = rnn_utils.pack_padded_sequence(X, X_lengths, batch_first=True)
        X, _ = self.lstm(X, hidden)
        # print("lstm_out shape:{}".format(X.shape))
        X, _ = rnn_utils.pad_packed_sequence(X, batch_first=True)
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        # Get the emission scores from the BiLSTM
        lstm_feats = self.hidden_to_tag(X)

        if run_mode == 'train':
            return lstm_feats
        elif run_mode == 'eval' or 'test':
            # print("lstm_feats shape:{}".format(lstm_feats.shape))
            # viterbi解码得到预测
            score, seqs_tag = self.crf._viterbi_decode(lstm_feats)
            return seqs_tag
        else:
            raise RuntimeError("Parameter 'run_mode' need a value.")

    def run_model(self, model, run_mode, data_path):
        # loss_function = self.criterion_dict[self.criterion_name]()
        optimizer = self.optimizer_dict[self.optimizer_name](model.parameters(), lr=self.learning_rate)

        if run_mode == 'train':
            print('Running {} model. Training...'.format(self.model_name))
            model.train()
            for epoch in range(self.num_epochs):
                for x, x_len, y, y_len in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                                   run_mode=run_mode):
                    batch_x = torch.LongTensor(x)
                    # print('batch_x shape: {}'.format(batch_x.shape))
                    model.zero_grad()
                    tag_scores = model(batch_x, x_len, run_mode=run_mode)
                    # print('tag_score shape: {}'.format(tag_scores.shape))
                    batch_y = torch.LongTensor(y)
                    batch_y = batch_y.view(-1)
                    loss = self.crf.neg_log_likelihood(tag_scores, batch_y)
                    # loss = loss_function(tag_scores, batch_y)
                    # print('Main Loss = {}'.format(loss))
                    loss.backward()
                    optimizer.step()

                print('Done epoch {} of {}'.format(epoch + 1, self.num_epochs))
            model_save_path = self.data_root + self.model_save_path
            torch.save(model, '{}'.format(model_save_path))
        elif run_mode == 'eval' or 'test':
            print('Running {} model. {}ing...'.format(self.model_name, run_mode))
            model.eval()
            with torch.no_grad():
                for x, x_len, y, y_len in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                                   run_mode=run_mode):
                    batch_x = torch.tensor(x).long()
                    y_predict = model(batch_x, x_len, run_mode)
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
        # y = list(torch.tensor(y, dtype=int).view(-1).numpy())
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
