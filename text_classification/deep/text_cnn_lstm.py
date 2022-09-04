#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
CNNLSTM
======
A class for something.
"""

import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_cnn import TextCNN


class CNNLSTM(TextCNN):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        # 调用父类BaseModel
        super(CNNLSTM, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                      dropout_rate, learning_rate, num_epochs, batch_size,
                                      criterion_name, optimizer_name, gpu, **kwargs)

        # RNN堆叠的层数，默认为1层
        self.num_layers = 1
        if 'num_layers' in kwargs:
            self.num_layers = kwargs['num_layers']
        # RNN的方向性，双向RNN则取值2；单向RNN则取1
        self.num_directions = 1
        if 'num_directions' in kwargs:
            self.num_directions = kwargs['num_directions']
        self.bidirectional = True if self.num_directions == 2 else False
        # 使用的神经网络类型
        self.rnn_model = 'LSTM'
        if 'rnn_model' in kwargs:
            self.rnn_model = kwargs['rnn_model']
        cnn_dim = self.num_filters * len(self.filter_sizes)

        # RNN模型初始化
        if self.rnn_model == 'LSTM':
            self.model = nn.LSTM(input_size=cnn_dim, hidden_size=hidden_dim, num_layers=self.num_layers,
                                 dropout=dropout_rate, batch_first=True, bidirectional=self.bidirectional)
        elif self.rnn_model == 'GRU':
            self.model = nn.GRU(input_size=cnn_dim, hidden_size=hidden_dim, num_layers=self.num_layers,
                                dropout=dropout_rate, batch_first=True, bidirectional=self.bidirectional)
        elif self.rnn_model == 'RNN':
            self.model = nn.RNN(input_size=cnn_dim, hidden_size=hidden_dim, num_layers=self.num_layers,
                                dropout=dropout_rate, batch_first=True, bidirectional=self.bidirectional)
        else:
            print('No such RNN model!')

        # 设置输出层的参数
        self.fc_out = nn.Linear(self.hidden_dim * self.num_directions, self.num_classes)

    # 模型的前向传播
    def forward(self, x):
        # input: [batch_size, seq_len]
        # 词嵌入
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        # 增加维度。由于卷积操作是在二维平面上进行的，而词向量内部不能拆分（拆分没有意义），因此需要增加一维
        cnn_in = embed.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        # 进行卷积和池化操作，并将结果按列（即维数1）拼接
        cnn_out = torch.cat([self.con_and_pool(cnn_in, con) for con in self.convs], 1)  # size同下
        cnn_out = self.drop_out(cnn_out).unsqueeze(1)  # [batch_size, 1, num_filters * len(filter_sizes)]
        rnn_out, _ = self.model(cnn_out)  # [batch_size, 1, hidden_dim * num_directions]
        out = self.fc_out(rnn_out.squeeze(1))  # [batch_size, num_classes]
        return out


# 程序入口
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')

        # 设置测试用例，验证模型是否能够运行
        # 设置模型参数
        vocab_size, embed_dim, hidden_dim, num_classes, \
        dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu, num_filters, filter_sizes, num_layers, num_directions \
            = 100, 64, 32, 2, 0.5, 0.001, 3, 32, 'CrossEntropyLoss', 'Adam', 0, 2, [1, 2, 3], 2, 2
        # 测试所用为2层、双向LSTM+CNN
        model = CNNLSTM(vocab_size, embed_dim, hidden_dim, num_classes,
                        dropout_rate, learning_rate, num_epochs, batch_size,
                        criterion_name, optimizer_name, gpu, num_filters=num_filters, filter_sizes=filter_sizes,
                        num_layers=num_layers, num_directions=num_directions)
        # 传入简单数据，查看模型运行结果
        # [batch_size, seq_len] = [3, 5]
        input_data = torch.LongTensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 4, 2, 7, 5]])
        output_data = model(input_data)
        print("The output is: {}".format(output_data))

        print("The test process is done.")

    else:
        print("There is no {} function. Please check your command.".format(args.phase))

    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done CNN LSTM.')