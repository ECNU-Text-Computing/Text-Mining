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
注意：①需要下载谷歌中文BERT预训练模型，代码中对应修改self.bert_path的保存路径；②增加testdata.txt文件作为测试数据
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
BERT的下游任务：MLM或NSP
    1、MLM：掩码语言模型
    2、NSP：句子连贯性判断
======
[MASK] ：表示这个词被遮挡。需要带着[]，并且mask是大写，对应的编码是103
[SEP]: 表示分隔开两个句子。对应的编码是102
[CLS]:用于分类场景，该位置可表示整句话的语义。对应的编码是101
[UNK]：文本中的元素不在词典中，用该符号表示生僻字。对应编码是100
[PAD]：针对有长度要求的场景，填充文本长度，使得文本长度达到要求。对应编码是0
"""

import argparse
import datetime
import torch
import torch.nn as nn
from base_model import BaseModel
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

# BERT文本分类
# padding符号, bert中综合信息符号
PAD, CLS = '[PAD]', '[CLS]'


class BERT(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(BERT, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                   dropout_rate, learning_rate, num_epochs, batch_size,
                                   criterion_name, optimizer_name, gpu, **kwargs)
        # 基本参数设置
        # 每句话处理成的长度(短填长切)
        self.pad_size = 10
        if 'pad_size' in kwargs:
            self.pad_size = kwargs['pad_size']
        # 若超过1000batch效果还没提升，则提前结束训练
        self.require_improvement = 1000
        # BERT预训练模型存储地址：
        self.bert_path = 'E:/GitRepo/Text_Mining/text_classification/deep/bert_pretrain'
        # BERT切分词
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert = BertModel.from_pretrained(self.bert_path)

        for param in self.bert.parameters():
            param.requires_grad = True
        
        # 输出层
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

    def load_dataset(self, path):
        pad_size = self.pad_size
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content = lin
                token = self.tokenizer.tokenize(content)
                seq_len = len(token)
                mask = []
                token_ids = self.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, seq_len, mask))
        return contents

    def forward(self, x, mask):
        # input x: [batch_size, seq_len]
        outputs = self.bert(x, attention_mask=mask)
        pooled = outputs[1]
        out = self.fc(pooled)

        # 用from pytorch_pretrained_bert时的输出
        # _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # output_all_encoded_layers=True时，12层Transformer的结果全返回，存在第一个列表中，
        # 每个encoder_output的大小为[batch_size, sequence_length, hidden_size]
        # pool_out = [batch_size, hidden_size]，取了最后一层Transformer的输出结果的第一个单词[cls]的hidden states，其已经蕴含了整个input句子的信息了
        # out = self.fc(pooled)

        return out


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
        vocab_size, embed_dim, hidden_dim, num_classes, \
        dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu = 200, 64, 768, 2, 0.5, 0.0001, 3, 128, 'CrossEntropyLoss', 'Adam', 0

        # new an objective.
        model = BERT(vocab_size, embed_dim, hidden_dim, num_classes,
                     dropout_rate, learning_rate, num_epochs, batch_size,
                     criterion_name, optimizer_name, gpu)
        # a simple example of the input.
        input = model.load_dataset(
            r'E:/GitRepo/Text_Mining/text_classification/deep/bert_pretrain/testdata.txt')
        # 应该输入文本的序列，用tokenizer变成id序列
        m = len(input)

        x = []  # 取第一行
        for i in range(m):
            x.append(input[i][0])
        x = torch.LongTensor(x)  # x: [batch_size, seq_len] = [3, 10]

        mask = []
        for i in range(m):
            mask.append(input[i][2])
        mask = torch.LongTensor(mask)  # mask: 掩码，每一个句子的长度，如[5,9,7]

        output = model(x, mask)
        print(output)
        print('The test process is done!')

    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done BERT Model!')
