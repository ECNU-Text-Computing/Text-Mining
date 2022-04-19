import torch.nn as nn
#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
TextLSTMAttention
======
A class for something.
"""
import os
import sys
import argparse
import datetime
import torch

import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from Deep.Base_Model import Base_Model
from utils.metrics import cal_all

# Bi-LSTM模型
class BiLSTAttention(Base_Model):
    def __init__(self, vocab_size, hidden_dim, embed_dim, num_classes,num_layers, num_directions,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs ):
        super(BiLSTAttention, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)

        self.input_size = embed_dim
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.num_directions = num_directions

        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=num_layers, bidirectional=(num_directions == 2))
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True)
        ) #传入有序模块
        self.liner = nn.Linear(hidden_dim, num_classes)
        self.act_func = nn.Softmax(dim=1)

    def forward(self, x):
        # lstm的输入维度为 [seq_len, batch, input_size]
        # x [batch_size, seq_len, embed_dim]
        x=self.embedding(x)
        #print(x.size())
        x = x.permute(1, 0, 2)  # [seq_len, batch_size,embed_dim]
        #print(x.size())

        # 由于数据集不一定是预先设置的batch_size的整数倍，所以用size(1)获取当前数据实际的batch
        batch_size = x.size(1)
        print(batch_size)
        # 设置lstm最初的前项输出,bi-lstm里面有隐藏层，bi-lstm和其他网络不同在于隐藏层初始化在前向传播处发生，要保证tensor在一个设备上
        h_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(self.device)
        c_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(self.device)
        #print(h_0.size())
        #print(c_0.size())
        # out[seq_len, batch, num_directions * hidden_size]。多层lstm，out只保存最后一层每个时间步t的输出h_t
        # h_n, c_n [num_layers * num_directions, batch, hidden_size]
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        #print(out.size())

        # 将双向lstm的输出拆分为前向输出和后向输出
        (forward_out, backward_out) = torch.chunk(out, 2, dim=2) # trunk在不同维度上切分
        out = forward_out + backward_out  # [seq_len, batch, hidden_size]
        #print(out.size())
        out = out.permute(1, 0, 2)  # [batch, seq_len, hidden_size]
        #print((out.size()))

        # 为了使用到lstm最后一个时间步时，每层lstm的表达，用h_n生成attention的权重
        h_n = h_n.permute(1, 0, 2)  # [batch, num_layers * num_directions,  hidden_size]
        #print(h_n.size())
        # h_n加和
        h_n = torch.sum(h_n, dim=1)  # [batch, 1,  hidden_size]
        #print(h_n.size())
        h_n = h_n.squeeze(dim=1)  # [batch, hidden_size]
        #print(h_n.size())
        #通过h_n得到attention权重
        attention_w = self.attention_weights_layer(h_n)  # [batch, hidden_size]
        #print(attention_w.size())
        attention_w = attention_w.unsqueeze(dim=1)  # [batch, 1, hidden_size]
        #print(attention_w.size())
        #权重内容
        attention_context = torch.bmm(attention_w, out.transpose(1, 2))  # [batch, 1, seq_len] #三维数组的乘法
        #print(attention_context.size())
        #权重归一化
        softmax_w = F.softmax(attention_context, dim=-1)  # [batch, 1, seq_len]
        #print(softmax_w.size())
        #每层的输出和attention权重相乘
        x = torch.bmm(softmax_w, out)  # [batch, 1, hidden_size]
        #print(x.size())
        x = x.squeeze(dim=1)  # [batch, hidden_size]
        #print(x.size())
        x = self.liner(x)
        #print(x.size())
        x = self.act_func(x)
        #print(x.size())
        return x

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    # python3 text_bilstmattention.py --phase test
    if args.phase == 'test':
        # For testing our model, we can set the hyper-parameters as follows.
        vocab_size, embed_dim, hidden_dim, num_classes, num_layers, num_directions,\
        dropout_rate, learning_rate, num_epochs, batch_size,\
        criterion_name, optimizer_name, gpu= 200, 64, 64, 2, 1,2,0.5, 0.0001, 3, 64, 'CrossEntropyLoss', 'Adam', 0

        # new an objective.
        model =BiLSTAttention(vocab_size, embed_dim, hidden_dim, num_classes,num_layers, num_directions,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu)
        # a simple example of the input.
        input = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
        input = torch.LongTensor(input)  # input: [batch_size, seq_len] = [3, 5]

        # the designed model can produce an output.
        output= model(input)
        print(output)
        print('This is a test process.')
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Base_Model!')
