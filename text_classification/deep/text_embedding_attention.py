"""
Attention
======
Attention计算方式可选：
    点积（dot product）
    串联（concat）
    余弦相似度（cosine similarity）
    多层感知机（multilayer perceptron）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import datetime
from deep.base_model import BaseModel


class Attention(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(Attention, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                        dropout_rate, learning_rate, num_epochs, batch_size,
                                        criterion_name, optimizer_name, gpu, **kwargs)
        self.tanh = nn.Tanh()
        # 设置采用何种方式计算注意力
        self.attention_name = 'dot product'
        if 'attention_name' in kwargs:
            self.attention_name = kwargs['attention_name']
        self.dim_q = self.embed_dim
        if 'dim_q' in kwargs:
            self.dim_q = kwargs['dim_q']
        self.dim_hidden = self.embed_dim
        if 'dim_hidden' in kwargs:
            self.dim_hidden = kwargs['dim_hidden']

        # 以内积方式取得注意力
        if self.attention_name == 'dot product':
            self.out_trans = nn.Linear(self.embed_dim, self.embed_dim)
            self.w = nn.Linear(self.embed_dim, 1, bias=False)
        # 以拼接方式取得注意力
        elif self.attention_name == 'concat':
            self.out_trans = nn.Linear(self.embed_dim, self.embed_dim + self.dim_q)
            self.w = nn.Linear(self.embed_dim + self.dim_q, 1, bias=False)
        # 以余弦相似度方式取得注意力
        elif self.attention_name == 'cosine similarity':
            self.out_trans = nn.Linear(self.embed_dim, self.embed_dim)
        # 以多层感知机方式取得注意力的
        elif self.attention_name == 'multilayer perceptron':
            self.out_trans = nn.Linear(self.embed_dim, self.dim_hidden)
            self.w = nn.Linear(self.dim_hidden, 1, bias=False)
        else:
            print('No such Attention model!')

    def forward(self, x):
        # input_data: [batch_size, seq_len]
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        M = self.out_trans(embed)  # [batch_size, seq_len, embed_dim]
        if self.attention_name == 'cosine similarity':
            # 计算余弦相似度后维度会减少一维，因此用unsqueeze函数补上
            M2 = F.cosine_similarity(embed, M, dim=2).unsqueeze(2)
            # 注意：进行softmax操作时一定要指定维度
            alpha = F.softmax(M2)
        elif self.attention_name == 'multilayer perceptron':
            M2 = self.out_trans(M)
            M2 = M2 + M
            alpha = F.softmax(self.w(self.tanh(M2)))
        else:
            alpha = F.softmax(self.w(M), dim=1)
            print(alpha.size())
        out = embed * alpha  # [batch_size, seq_len, embed_dim]
        out = torch.sum(out, dim=1)  # [batch_size, embed_dim]
        out = F.relu(out)
        # 通过两个全连接层后输出
        out = self.fc(out)
        out = self.fc_out(out)
        return out


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, \
        num_epochs, batch_size, criterion_name, optimizer_name, gpu, attention_name \
            = 100, 64, 32, 2, 0.5, 0.0001, 3, 64, 'CrossEntropyLoss', 'Adam', 0, 'cosine similarity'

        model = Attention(vocab_size, embed_dim, hidden_dim, num_classes,
                          dropout_rate, learning_rate, num_epochs, batch_size,
                          criterion_name, optimizer_name, gpu, attention_name=attention_name)

        input = torch.LongTensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
                                  [1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 4, 8, 3, 6]])  # [batch_size, seq_len] = [6, 5]
        output = model(input)
        print(output)

        print('The test process is done.')

    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Attention!')
