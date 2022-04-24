#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
BERT
======
A class for something.
======
参考文献：
    1、https://blog.csdn.net/weixin_42237487/article/details/112355703
    2、https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch

注意：需要下载谷歌中文BERT预训练模型，代码中对应修改self.bert_path的保存路径

BERT特点：
    1、BERT可以解决词的情态表征，解决一词多义问题。
        Bert生成动态Word Embedding的思路：
        事先用一个学到的单词的Word Embedding，该Word Embedding在使用的时候已经具备了特定的上下文意思了，可以根据上下文单词的语义去调整（自注意力）单词的Word Embedding表示，
        经过调整后的Word Embedding更能表达在这个上下文中的具体含义，自然也就解决了多义词的问题了。
    2、Bert编码器（特征提取）最终可以输出两个Embedding信息：单词的Word Embedding和句子的Embedding。
    3、可以提取出Bert编码器输出的句子Embedding，经过全连接的网络层对句子进行情感判断。这个Embedding携带了当前句子的主要信息。

模型由输入层、编码层和输出层三个部分组成
BERT的输入：input
    1、Token Embedding：词特征（词向量）的嵌入，针对中文，目前只支持字特征嵌入
    2、Segment Embedding：词的句子级特征嵌入，针对双句子输入任务，做句子A，B嵌入，针对单句子任务，只做句子A嵌入
    3、Position Embedding：词的位置特征，针对中文，目前最大长度为 512

LSTM的输出：MLM或NSP
    1、ontput(seq_len, batch, num_directions * hidden_size)保存了最后一层，每个time_step的输出,最后一项output[:, -1, :]
    2、hn(num_layers * num_directions, batch, hidden_size)最后时刻的隐藏状态
    3、cn(num_layers * num_directions, batch, hidden_size)最后时刻的单元状态，一般用不到
======
"""

import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.metrics import cal_all
from Deep.Base_Model import Base_Model
from pytorch_pretrained_bert import BertModel, BertTokenizer

#BERT文本分类

class BERT(Base_Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(BERT, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)

        self.require_improvement = 1000                 # 若超过1000batch效果还没提升，则提前结束训练
        self.pad_size = 32                              # 每句话处理成的长度(短填长切),测试文件时没法调用config文件，所以直接设置
        #self.pad_size = kwargs['pad_size']             # 可从config文件里设置
        self.bert_path = 'F:/2-PostGraduate/2022.3.1Text-Mining/code/Text_Mining-20220329/Text_Mining/Deep/bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)   # bert切分词

        self.bert = BertModel.from_pretrained(self.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(self.hidden_dim, self.num_classes)   #直接分类

        # self.output = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(768 + embed_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, num_classes)
        # )

    def forward(self, x):
         context = x  # 输入的句子
         mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
         mask = torch.unsqueeze(mask, 0)
         _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
         #output_all_encoded_layers=True时，12层Transformer的结果全返回，存在第一个列表中，每个encoder_output的大小为[batch_size, sequence_length, hidden_size]
         #pool_out = [batch_size, hidden_size]，取了最后一层Transformer的输出结果的第一个单词[cls]的hidden states，其已经蕴含了整个input句子的信息了
         out = self.fc(pooled)

         return out

    # def forward(self, seqs, features):
    #     _, pooled = self.bert(seqs, output_all_encoded_layers=False)
    #     concat = torch.cat([pooled, features], dim=1)
    #     logits = self.output(concat)
    #     return logits


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()
    print()
    #python3 text_bert.py --phase test
    if args.phase == 'test':
        print('This is a test process.')
        # For testing our model, we can set the hyper-parameters as follows.
        vocab_size, embed_dim, hidden_dim, num_classes,  \
        dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu = 200, 64, 768, 2, 0.5, 0.0001, 3, 128, 'CrossEntropyLoss', 'Adam', 0

        # new an objective.
        model = BERT(vocab_size, embed_dim, hidden_dim, num_classes,
                               dropout_rate, learning_rate, num_epochs, batch_size,
                               criterion_name, optimizer_name, gpu)
        # a simple example of the input.
        input = [[1, 2, 3, 4, 0], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
        input = torch.LongTensor(input)  # input: [batch_size, seq_len] = [3, 5]

        # the designed model can produce an output.
        output = model(input)
        print(output)
        print('Done!')

    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done BERT_Model!')


