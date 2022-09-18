#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
BERT2(与text_bert.py中的BERT模型区别：本模型接受的是BertTokenizer处理后的数据)
======
参考文献：
    1. https://blog.csdn.net/weixin_42237487/article/details/112355703
    2. https://huggingface.co/docs/transformers/v4.21.1/en/model_doc/bert#transformers.BertModel
BERT特点：
    1. BERT可以解决词的情态表征，解决一词多义问题。
        Bert生成动态Word Embedding的思路：
        事先用一个学到的单词的Word Embedding，该Word Embedding在使用的时候已经具备了特定的上下文意思了，
        可以根据上下文单词的语义去调整（自注意力）单词的Word Embedding表示，
        经过调整后的Word Embedding更能表达在这个上下文中的具体含义，自然也就解决了多义词的问题了。
    2. Bert编码器（特征提取）最终可以输出两个Embedding信息：单词的Word Embedding和句子的Embedding。
    3. 可以提取出Bert编码器输出的句子Embedding，经过全连接的网络层对句子进行情感判断。这个Embedding携带了当前句子的主要信息。
模型由输入层、编码层和输出层三个部分组成
BERT的输入：input_data
    1. Token Embedding：词特征（词向量）的嵌入，针对中文，目前只支持字特征嵌入
    2. Segment Embedding：词的句子级特征嵌入，针对双句子输入任务，做句子A，B嵌入，针对单句子任务，只做句子A嵌入
    3. Position Embedding：词的位置特征，针对中文，目前最大长度为 512
BERT的输出：
    1. last_hidden_state：torch.FloatTensor类型，最后一个隐藏层的序列的输出，形状为[batch_size, seq_len, hidden_size]
    2. pooler_output：torch.FloatTensor类型，[CLS]的这个token的输出，形状为[batch_size, hidden_size]
    3. hidden_states(可选项)：tuple(torch.FloatTensor)类型。需要指定config.output_hidden_states=True。
    它是一个元组，第一个元素是embedding，其余元素是各层的输出，每个元素的形状为[batch_size, seq_len, hidden_size]
    4. attentions(可选项)：tuple(torch.FloatTensor)类型。需要指定config.output_attentions=True。
    它也是一个元组，它的元素是每一层的注意力权重，用于计算self-attention heads的加权平均值，形状为[batch_size, seq_len, hidden_size]
BERT的下游任务：MLM或NSP
    1、MLM：掩码语言模型
    2、NSP：句子连贯性判断
======
[UNK]：文本中的元素不在词典中，用该符号表示生僻字。对应编码是100
[CLS]:用于分类场景，该位置可表示整句话的语义。对应的编码是101
[SEP]: 表示分隔开两个句子。对应的编码是102
[MASK] ：表示这个词被遮挡。需要带着[]，并且mask是大写，对应的编码是103
[PAD]：针对有长度要求的场景，填充文本长度，使得文本长度达到要求。对应编码是0
"""

import argparse
import datetime
import torch
import torch.nn as nn
from transformers import BertModel

from deep.base_model import BaseModel


# BERT文本分类
class BERT2(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(BERT2, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                    dropout_rate, learning_rate, num_epochs, batch_size,
                                    criterion_name, optimizer_name, gpu, **kwargs)
        # 一些重要的参数，这些参数几乎在所有的深度学习模型中都会被用到。

        self.vocab_size = vocab_size  # 词表大小
        self.embed_dim = embed_dim  # 词嵌入/向量的维度
        self.hidden_dim = 768  # 隐藏层的维度，bert模型的输出向量为768维
        self.num_classes = num_classes  # 类别数量
        self.dropout_rate = dropout_rate  # dropout的比率，即丢弃神经网络中的多少神经元（向量中的元素）
        self.learning_rate = learning_rate  # 优化器中的学习率
        self.num_epochs = num_epochs  # 要将数据迭代训练多少轮
        self.batch_size = batch_size  # 每次训练要喂给（feed）模型多少样本
        self.criterion_name = criterion_name  # 损失函数
        self.optimizer_name = optimizer_name  # 优化器

        # 加载的预训练模型名称
        self.model_path = 'bert-base-uncased'
        if 'model_path' in kwargs:
            self.model_path = kwargs['model_path']
        # 隐藏层维度设置
        self.hidden_dim2 = self.hidden_dim // 2
        if 'hidden_dim2' in kwargs:
            self.hidden_dim2 = kwargs['hidden_dim2']
        # 设置模型评价指标数量
        self.metrics_num = 4
        if 'metrics_num' in kwargs:
            self.metrics_num = kwargs['metrics_num']

        # bert模型设置
        self.bert = BertModel.from_pretrained(self.model_path, return_dict=True, output_attentions=True,
                                              output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad = True
        # 根据输出层的不同复杂程度，设置不同模式
        # 默认为普通模式，仅通过一个全连接层
        self.mode = 'normal'
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        if 'mode' in kwargs:
            if kwargs['mode'] == 'adap':
                # 进阶模式，通过两个全连接层后输出
                print('Using adap mode.')
                self.mode = 'adap'
                self.fc = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim2),
                    nn.Tanh(),
                    nn.Linear(self.hidden_dim2, self.num_classes)
                )
            elif kwargs['mode'] == 'pro':
                # 高级模式，通过三个全连接层后输出
                print('Using pro mode.')
                self.fc = nn.Sequential(
                    nn.Linear(self.hidden_dim, int(self.hidden_dim / 2)),
                    # nn.ReLU(),
                    nn.Tanh(),
                    nn.Dropout(p=self.dropout_rate),
                    nn.Linear(int(self.hidden_dim / 2), int(self.hidden_dim / 4)),
                    # nn.ReLU(),
                    nn.Tanh(),
                    nn.Dropout(p=self.dropout_rate),
                    nn.Linear(int(self.hidden_dim / 4), self.num_classes)
                )
        else:
            print('Using normal mode.')
        # weights matrix =  [hidden_dim, num_classes] = [64, 2]
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.drop_out = nn.Dropout(dropout_rate)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # input: [batch_size, seq_len]
        output = self.bert(x)['last_hidden_state']
        out = torch.mean(output, dim=1)  # [batch_size, hidden_dim]
        out = self.drop_out(out)  # [batch_size, num_classes]
        out = self.fc(out)  # [batch_size, num_classes]

        # 用from pytorch_pretrained_bert时的输出
        # _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # output_all_encoded_layers=True时，12层Transformer的结果全返回，存在第一个列表中，
        # 每个encoder_output的大小为[batch_size, sequence_length, hidden_dim]
        # pool_out = [batch_size, hidden_dim]，取了最后一层Transformer的输出结果的第一个单词[cls]的hidden states，其已经蕴含了整个input句子的信息了
        # out = self.fc(pooled)

        return out


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
        vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu, mode \
            = 100, 768, 768, 2, 0.5, 0.0001, 3, 8, 'CrossEntropyLoss', 'Adam', 0, 'pro'

        # 测试所用为pro模式
        model = BERT2(vocab_size, embed_dim, hidden_dim, num_classes,
                      dropout_rate, learning_rate, num_epochs, batch_size,
                      criterion_name, optimizer_name, gpu, mode=mode)
        # 测试数据为id时
        '''
        input_data = torch.LongTensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
                                       [1, 3, 5, 7, 9], [2, 4, 6, 8, 10],
                                       [1, 4, 8, 3, 6]])  # [batch_size, seq_len] = [6, 5]
        output_data = model(input_data)
        '''
        # 测试数据为文本时
        with open(r'E:\GitRepo\Text-Mining\text_classification\deep\test_data.txt', 'r') as fp:
            input_data = fp.readlines()
        output_data = model(input_data)
        print(output_data)
        print('The test process is done!')

    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done BERT2 Model!')
