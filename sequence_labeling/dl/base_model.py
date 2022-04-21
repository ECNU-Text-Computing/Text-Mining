#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Base_Model
======
A class for BaseModel for all basic deep learning models.
配置文件：cmed.dl.base_model.norm.json
forward(X, X_lengths)：
    X：padding后的输入数据。2维列表，size = [batch_size, seq_len]
    X_lengths: 输入数据的真实长度。1维列表，size = [batch_size]
    返回值：tag_scores。2维张量，shape = [batch_size*seq_len, tags_size]
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
        self.model_name = config['model_name']
        self.data_root = config['data_root']
        self.model_save_path = config['model_save_name']

        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.layers = config['num_layers']
        self.bidirectional = True if config['bidirectional'] == 'True' else False
        self.n_directions = 2 if self.bidirectional else 1
        self.dropout_p = config['dropout_rate']
        self.pad_token = config['pad_token']

        self.num_epochs = config['num_epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']

        self.criterion_dict = {
            'NLLLoss': torch.nn.NLLLoss,
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
        }
        self.criterion_name = config['criterion_name']
        if self.criterion_name in self.criterion_dict:
            self.criterion = self.criterion_dict[self.criterion_name]()

        self.optimizer_dict = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam
        }
        self.optimizer_name = config['optimizer_name']
        if self.optimizer_name not in self.optimizer_dict:
            raise ValueError("There is no optimizer_name: {}.".format(self.optimizer_name))

        self.tag_index_dict = DataProcessor(**self.config).load_tags()
        self.tags_size = len(self.tag_index_dict)
        vocab = DataProcessor(**self.config).load_vocab()
        vocab_size = len(vocab)
        self.padding_idx = vocab[self.pad_token]
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim,
                                            padding_idx=self.padding_idx)

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.layers,
                            bidirectional=self.bidirectional, batch_first=True)

        # 将模型输出映射到标签空间
        self.output_to_tag = nn.Linear(self.hidden_dim * self.n_directions, self.tags_size)

        print('完成类{}的初始化'.format(self.__class__.__name__))

    def forward(self, X, X_lengths):
        batch_size, seq_len = X.size()
        embeded = self.word_embeddings(X)
        embeded = rnn_utils.pack_padded_sequence(embeded, X_lengths, batch_first=True)
        output, _ = self.lstm(embeded)  # [batch_size, seq_len, hidden_dim]
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        out = output.contiguous()  # contiguous()执行强制拷贝,断开两个变量之间的依赖
        out = out.view(-1, out.shape[2])  # 降维为[batch_size*seq_len, hidden_dim]
        out = self.output_to_tag(out)  # [batch_size*seq_len, tags_size]

        tag_scores = F.log_softmax(out, dim=1)  # [batch_size*seq_len, tags_size]

        return tag_scores

    def run_train(self, model):
        print('Running {} model. Training...'.format(self.model_name))
        run_mode = 'train'
        optimizer = self.optimizer_dict[self.optimizer_name](model.parameters(), lr=self.learning_rate)
        model.train()

        train_losses = []  # 记录每一batch的loss
        valid_losses = []
        best_valid_loss = float('inf')
        for epoch in range(self.num_epochs):
            train_loss = 0
            train_data_num = 0
            for x, x_len, y, y_len in DataLoader(**self.config).data_generator(data_path=self.data_root,
                                                                               run_mode=run_mode):
                train_data_num += len(x)
                batch_x = torch.tensor(x).long()
                # print('batch_x shape: {}'.format(batch_x.shape))
                model.zero_grad()
                tag_scores = model(batch_x, x_len)

                batch_y = torch.tensor(y).long()
                batch_y = batch_y.view(-1)
                # print('batch_y shape: {}'.format(batch_y.shape))
                loss = self.criterion(tag_scores, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_losses.append(train_loss)  # 记录评价结果

            # 模型验证
            eval_loss = self.evaluate(model)
            print("Done Epoch{}. Eval_Loss={}".format(epoch + 1, eval_loss))
            valid_losses.append(eval_loss)

            # 保存参数最优的模型
            if eval_loss < best_valid_loss:
                best_valid_loss = eval_loss
                model_save_path = '{}{}_{}'.format(self.data_root, self.model_name, self.model_save_path)
                torch.save(model, '{}'.format(model_save_path))

        self.prtplot(train_losses)
        self.prtplot(valid_losses)

    def evaluate(self, model):
        # print('Running {} model. Evaluating...'.format(self.model_name))
        run_mode = 'eval'
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            eval_data_num = 0
            for x, x_len, y, y_len in DataLoader(**self.config).data_generator(data_path=self.data_root,
                                                                               run_mode=run_mode):
                eval_data_num += len(x)
                batch_x = torch.tensor(x).long()
                tag_scores = model(batch_x, x_len)

                batch_y = torch.tensor(y).long()
                batch_y = batch_y.view(-1)
                loss = self.criterion(tag_scores, batch_y)
                eval_loss += loss.item()

        return eval_loss

    def test(self):
        print('Running {} model. Testing...'.format(self.model_name))
        run_mode = 'test'
        best_model_path = '{}{}_{}'.format(self.data_root, self.model_name, self.model_save_path)
        model = torch.load(best_model_path)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            test_data_num = 0
            for x, x_len, y, y_len in DataLoader(**self.config).data_generator(data_path=self.data_root,
                                                                               run_mode=run_mode):
                test_data_num += len(x)
                batch_x = torch.tensor(x).long()
                tag_scores = model(batch_x, x_len)

                batch_y = torch.tensor(y).long()
                batch_y = batch_y.view(-1)
                loss = self.criterion(tag_scores, batch_y)
                test_loss += loss.item()

                # 返回每一行中最大值的索引
                predict = torch.max(tag_scores, dim=1)[1]  # 1维张量
                y_predict = list(predict.numpy())
                y_predict = self.index_to_tag(y_predict)
                print(y_predict)

                y_true = y.flatten()
                y_true = self.index_to_tag(y_true)
                print(y_true)

                # 输出评价结果
                print(Evaluator().classifyreport(y_true, y_predict))

    def index_to_tag(self, y):
        index_tag_dict = dict(zip(self.tag_index_dict.values(), self.tag_index_dict.keys()))
        y_tagseq = []
        for i in range(len(y)):
            y_tagseq.append(index_tag_dict[y[i]])
        return y_tagseq

    def prtplot(self, y_axis_values):
        x_axis_values = np.arange(1, len(y_axis_values) + 1, 1)
        y_axis_values = np.array(y_axis_values)
        plt.plot(x_axis_values, y_axis_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.show()


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
