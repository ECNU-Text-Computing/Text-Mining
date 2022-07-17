#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
LSTM、BiLSTM
======
A class for something.
======
LSTM是RNN的一种变体，它对RNN进行了改进并有效避免了常规RNN梯度爆炸或梯度消失的问题

LSTM的输入
    1、input(seq_len, batch, input_size)
    2、h0(num_layers * num_directions, batch, hidden_size)初始的隐藏状态
    3、c0(num_layers * num_directions, batch, hidden_size)初始的单元状态

LSTM的输出
    1、ontput(seq_len, batch, num_directions * hidden_size)保存了最后一层，每个time_step的输出,最后一项output[:, -1, :]
    2、hn(num_layers * num_directions, batch, hidden_size)最后时刻的隐藏状态
    3、cn(num_layers * num_directions, batch, hidden_size)最后时刻的单元状态，一般用不到
======
"""

import datetime
import argparse
import torch
import torch.nn as nn
from deep.base_model import BaseModel

class TextLSTM(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        # 继承父类BaseModel的属性
        super(TextLSTM, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                       dropout_rate, learning_rate, num_epochs, batch_size,
                                       criterion_name, optimizer_name, gpu, **kwargs)

        # LSTM堆叠的层数，默认为2层
        self.num_layers = 2
        if 'num_layers' in kwargs:
            self.num_layers = kwargs['num_layers']
        # LSTM的方向性，双向LSTM则取值2；单向LSTM则取1
        self.num_directions = 2
        if 'num_directions' in kwargs:
            self.num_directions = kwargs['num_directions']
        self.bidirection = True if self.num_directions == 2 else False

        # 设置LSTM模型的参数
        self.lstm = nn.LSTM(embed_dim, hidden_dim, self.num_layers,
                            batch_first=True, dropout=dropout_rate, bidirectional=self.bidirection)
        # 设置输出层的参数
        self.fc_out = nn.Linear(hidden_dim * self.num_directions, num_classes)

    # 模型的前向传播
    def forward(self, x):
        # batch_x: [batch_size, seq_len]
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        hidden, _ = self.lstm(embed)  # [batch_size, seq_len, hidden_dim * self.num_directions]
        # -1表示取该维度从后往前的第一个，即只需要最后一个词的隐藏状态作为输出
        hidden = self.dropout(hidden[:, -1, :])  # [batch_size, hidden_dim * self.num_directions]
        out = self.fc_out(hidden)  # [batch_size, num_classes]
        return out

# 程序入口
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description="Process some description.")
    parser.add_argument('--phase', default='test', help='the function name')
    args = parser.parse_args()

    if args.phase == 'test':
        print("This is a test process.")

        # 设置测试用例，验证模型是否能够运行
        # 设置模型参数
        vocab_size, embed_dim, hidden_dim, num_classes, \
        dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu = 100, 64, 64, 2, 0.5, 0.001, 3, 32, 'CrossEntropyLoss', 'Adam', 0
        # 创建类的实例
        model = TextLSTM(vocab_size, embed_dim, hidden_dim, num_classes,
                         dropout_rate, learning_rate, num_epochs, batch_size,
                         criterion_name, optimizer_name, gpu)
        # 传入简单数据，查看模型运行结果
        input_data = torch.LongTensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 4, 2, 7, 5]])
        output_data = model(input_data)
        print("The output is: {}".format(output_data))

        print("The test process is done.")

    else:
        print("There is no {} function.".format(args.phase))

    end_time = datetime.datetime.now()
    print("{} takes {} seconds.".format(args.phase, (end_time-start_time).seconds))
    print("Done Text_LSTM.")



