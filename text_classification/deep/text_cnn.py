#! /user/bin/evn python
# -*- coding:utf8 -*-

import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import BaseModel


class TextCNN(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, num_filters, filter_sizes, **kwargs):
        # 调用父类BaseModel
        super(TextCNN, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                      dropout_rate, learning_rate, num_epochs, batch_size,
                                      criterion_name, optimizer_name, gpu, **kwargs)

        # 卷积核的数量
        self.num_channels = num_filters
        # 卷积核的大小。数据类型为列表，列表长度即为卷积核数量
        self.filter_sizes = filter_sizes
        # 设置CNN模型的参数。由于可能存在多个卷积核，因此需构建多个模型，此处采用list存储不同的模型
        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_channels, (k, self.embed_dim)) for k in self.filter_sizes])
        # 设置全连接层的参数
        self.fc1 = nn.Linear(self.num_channels * len(self.filter_sizes), hidden_dim)
        # dropout层、输出层已在BaseModel中定义，此处直接继承

    # 卷积及池化过程
    def con_and_pool(self, x, conv):
        # 卷积，随后去掉额外增加的维度
        x = F.relu(conv(x)).squeeze()  # [batch_size, num_channels, seq_len - filter_size + 1]
        # 池化，此处采用了最大池化（Max Pooling）
        x = F.max_pool1d(x, x.size(2)).squeeze()  # [batch_size, num_channels]
        return x

    # 模型的前向传播
    def forward(self, x):
        # batch_x: [batch_size, seq_len]
        # 词嵌入
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        # 增加维度。由于卷积操作是在二维平面上进行的，而词向量内部不能拆分（拆分没有意义），因此需要增加一维
        embed = embed.unsqueeze(1)  # [batch_size, 1, seq_len, embedding]
        # 进行卷积和池化操作，并将结果按列（即维数1）拼接
        cnn_out = torch.cat([self.con_and_pool(embed, con) for con in self.convs], 1)  # size同下
        cnn_out = self.dropout(cnn_out)  # [batch_size, num_channels * num_filters]
        hidden = self.fc1(cnn_out)  # [batch_size, hidden_dim]
        out = self.fc_out(hidden)  # [batch_size, num_classes]
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
        criterion_name, optimizer_name, gpu, num_filters, filter_sizes \
            = 100, 64, 32, 2, 0.5, 0.001, 3, 32, 'CrossEntropyLoss', 'Adam', 0, 2, [1, 2, 3]
        # 创建类的实例
        model = TextCNN(vocab_size, embed_dim, hidden_dim, num_classes,
                        dropout_rate, learning_rate, num_epochs, batch_size,
                        criterion_name, optimizer_name, gpu, num_filters, filter_sizes)
        # 传入简单数据，查看模型运行结果
        input_data = torch.LongTensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 4, 2, 7, 5]])
        output_data = model(input_data)
        print("The output is: {}".format(output_data))

        print("The test process is done.")

    else:
        print("There is no {} function. Please check your command.".format(args.phase))

    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Text_CNN.')
