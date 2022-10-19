#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: CRF_Series_BERT
======
配置文件：cmed.dl.crf_series_bert.norm.json
"""
import argparse
import datetime
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sequence_labeling.dl.crf_nn_bert import CRF_NN_BERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class CRF_Series_BERT(CRF_NN_BERT):
    def __init__(self, **config):
        super().__init__(**config)

        # 基础模型
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.relu = nn.ReLU()
        # MLP
        self.fc = nn.Sequential(nn.Linear(self.embedding_dim, self.hidden_dim), self.relu, self.dropout)
        self.linear_output_to_tag = nn.Linear(self.hidden_dim, self.tags_size)
        self.linear_output_to_hidden = nn.Linear(self.hidden_dim, self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.layers,
                            bidirectional=self.bidirectional, batch_first=True)
        self.rnn_output_to_tag = nn.Linear(self.hidden_dim * self.n_directions, self.tags_size)
        self.rnn_output_to_hidden = nn.Linear(self.hidden_dim * self.n_directions, self.hidden_dim)

        # CNN
        self.window_sizes = config['window_sizes']
        self.out_channels = config['out_channels']

        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.hidden_dim * self.n_directions,
                                    out_channels=self.out_channels,
                                    kernel_size=h, padding=(int((h - 1) / 2))),
                          # nn.BatchNorm1d(num_features=self.out_channels),
                          nn.ReLU())
            for h in self.window_sizes
        ])
        self.cnn_output_to_tag = nn.Linear(in_features=self.out_channels * len(self.window_sizes),
                                           out_features=self.tags_size)
        self.cnn_output_to_hidden = nn.Linear(self.out_channels * len(self.window_sizes), self.hidden_dim)

        # Multiheads Self-Attention
        self.nums_head = config['nums_head']
        self.input_dim = self.out_channels * len(self.window_sizes)
        self.dim_k = self.input_dim
        self.dim_v = self.input_dim
        assert self.dim_k % self.nums_head == 0
        assert self.dim_v % self.nums_head == 0
        # 定义WQ、WK、WV矩阵
        self.q = nn.Linear(self.input_dim, self.dim_k)
        self.k = nn.Linear(self.input_dim, self.dim_k)
        self.v = nn.Linear(self.input_dim, self.dim_v)
        self._norm_fact = 1 / math.sqrt(self.dim_k)

        self.att_output_to_tag = nn.Linear(self.input_dim, self.tags_size)
        self.att_output_to_hidden = nn.Linear(self.dim_v, self.hidden_dim)

        # parallel拼接各模型输出后进行全连接降维
        self.concat_output_to_tag = nn.Linear(
            (self.hidden_dim + self.hidden_dim * self.n_directions + self.out_channels * len(
                self.window_sizes) + self.dim_v), self.tags_size)
        self.multi_output_to_tag = nn.Linear(3 * self.hidden_dim, self.tags_size)

        # print('完成类{}的初始化'.format(self.__class__.__name__))

    def _lstm_init_hidden(self, batch_size):
        hidden = (torch.zeros(self.layers * self.n_directions, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.layers * self.n_directions, batch_size, self.hidden_dim).to(device))
        return hidden

    def _init_hidden(self, batch_size):
        return torch.zeros(self.layers * self.n_directions, batch_size, self.hidden_dim).to(device)

    def forward(self, seq_list):
        # BertModel embedding
        batch = self.tokenizer(seq_list, padding=True, truncation=True, return_tensors="pt").to(device)
        embedded = self.get_token_embedding(batch, 0)
        embedded = self.del_special_token(seq_list, embedded)  # [batch_size, seq_len, embed_dim]

        # BiLSTM
        batch_size = len(seq_list)
        hidden = self._lstm_init_hidden(batch_size)
        bilstm_output, _ = self.lstm(embedded, hidden)  # [batch_size, seq_len, embed_dim * n_directions]
        # print('lstm shape: {}'.format(bilstm_output.shape))

        # CNN
        cnn_embedded = bilstm_output
        # [batch_size, seq_len, embedding_dim]  -> [batch_size, embedding_dim, seq_len]
        cnn_embedded = cnn_embedded.permute(0, 2, 1)
        cnn_output = [conv(cnn_embedded) for conv in self.convs]  # out[i]: [batch_size, self.out_channels, 1]
        cnn_output = torch.cat(cnn_output, dim=1)  # 对应第⼆个维度（⾏）拼接起来，⽐如说5*2*1,5*3*1的拼接变成5*5*1
        cnn_output = cnn_output.transpose(1, 2)
        # print('cnn shape: {}'.format(cnn_output.shape))

        # MultiHeads self-attention
        att_embedded = cnn_output
        Q = self.q(att_embedded).reshape(-1, att_embedded.shape[0], att_embedded.shape[1], self.dim_k // self.nums_head)
        K = self.k(att_embedded).reshape(-1, att_embedded.shape[0], att_embedded.shape[1], self.dim_k // self.nums_head)
        V = self.v(att_embedded).reshape(-1, att_embedded.shape[0], att_embedded.shape[1], self.dim_v // self.nums_head)

        atten = nn.Softmax(dim=-1)(torch.matmul(Q, K.permute(0, 1, 3, 2)))  # Q * K.T() # batch_size * seq_len * seq_len

        att_output = torch.matmul(atten, V).reshape(att_embedded.shape[0], att_embedded.shape[1],
                                                    -1)  # Q * K.T() * V # batch_size * seq_len * dim_v
        # print('att shape: {}'.format(att_output.shape))
        att_output = self.dropout(att_output)
        output = self.att_output_to_tag(att_output)

        tag_scores = output.reshape(-1, output.shape[2])

        crf_score, seqs_tag = self.viterbi_decode(tag_scores)

        return tag_scores, crf_score, seqs_tag


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
