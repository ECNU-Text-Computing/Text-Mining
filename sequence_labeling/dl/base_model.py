#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Base_Model
======
A class for something.
"""
import argparse
import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
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


class BaseModel(nn.Module):

    def __init__(self, **config):
        super(BaseModel, self).__init__()
        self.config = config
        self.data_root = config['data_root']
        self.model_save_path = config['model_save_name']

        self.pad_token = config['pad_token']

        self.model_name = config['model_name']
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.tagset_size = config['tag_size']
        self.num_epochs = config['num_epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']

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
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, 1, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden_to_tag = nn.Linear(self.hidden_dim, self.tagset_size)

    def forward(self, X, X_lengths, run_mode):
        batch_size, seq_len = X.size()
        X = self.word_embeddings(X)
        X = rnn_utils.pack_padded_sequence(X, X_lengths, batch_first=True)
        X, _ = self.lstm(X)
        X, _ = rnn_utils.pad_packed_sequence(X, batch_first=True)  # X的shape为[batch_size, seq_len, hidden_dim]
        X = X.contiguous()
        X = X.view(-1, X.shape[2])  # 将X降维为[batch_size*seq_len, hidden_dim]
        X = self.hidden_to_tag(X)  # X的shape为[batch_size*seq_len, tagset_size]

        if run_mode == 'train':
            tag_scores = F.log_softmax(X, dim=1)  # tag_scores的shape为[batch_size*seq_len, tagset_size]
            # print('shape of tag_scores:{}'.format(tag_scores.shape))
            return tag_scores
        elif run_mode == 'eval' or 'test':
            tag_scores = F.log_softmax(X, dim=1)
            # print('标注结果转换为Tag索引序列：', torch.max(scores, dim=1))
            predict = list(torch.max(tag_scores, dim=1)[1].numpy())  # [batch_size, seq_len]大小的列表
            return predict
        else:
            raise RuntimeError("main.py调用model.run_model()时，参数'run_mode'未赋值！")


    def run_model(self, model, run_mode, data_path):
        loss_function = self.criterion_dict[self.criterion_name]()
        optimizer = self.optimizer_dict[self.optimizer_name](model.parameters(), lr=self.learning_rate)

        if run_mode == 'train':
            print('Running {} model. Training...'.format(self.model_name))
            model.train()
            acc_list = []  # 记录每一batch的acc
            for epoch in range(self.num_epochs):
                total_loss = 0
                batch_counter = 0  # batch计数器
                train_data_num = 0  # 数据计数器
                for x, x_len, y, y_len in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                                   run_mode=run_mode):
                    batch_x = torch.tensor(x).long()
                    print('batch_x shape: {}'.format(batch_x.shape))
                    model.zero_grad()
                    tag_scores = model(batch_x, x_len, run_mode)
                    batch_y = torch.tensor(y).long()
                    batch_y = batch_y.view(-1)
                    print('batch_y shape: {}'.format(batch_y.shape))
                    loss = loss_function(tag_scores, batch_y)
                    print('loss = {}'.format(loss))
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    batch_counter += 1
                    train_data_num += len(x)
                    if batch_counter % 5 == 0:
                        print("Done Epoch{}. Loss={}".format(epoch+1, total_loss / train_data_num))

                    y_predict = list(torch.max(tag_scores, dim=1)[1].numpy())
                    y_predict = self.index_to_tag(y_predict)
                    y_true = y.flatten()
                    y_true = self.index_to_tag(y_true)
                    acc_value = Evaluator().acc(y_true, y_predict)
                    acc_list.append(acc_value)  # 记录评价结果
                    print('acc_value = {}'.format(acc_value))
            n_batch = np.arange(1, len(acc_list) + 1, 1)
            acc_list = np.array(acc_list)
            plt.plot(n_batch, acc_list)
            plt.xlabel('Batch')
            plt.ylabel('Accuracy')
            plt.grid()
            plt.show()

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
