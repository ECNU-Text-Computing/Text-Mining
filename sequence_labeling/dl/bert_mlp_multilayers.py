#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: Bert_LSTM
======
A class for LSTM using Bert embedding.
配置文件：cmed.dl.bert_lstm.norm.json
"""
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from sequence_labeling.dl.bert_mlp import Bert_MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class Bert_MLP_Multilayers(Bert_MLP):
    def __init__(self, **config):
        super().__init__(**config)

        self.fc1 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), self.dropout)

    def forward(self, seq_list):
        batch = self.tokenizer(seq_list, padding=True, truncation=True, return_tensors="pt").to(device)
        embedded = self.get_token_embedding(batch, 0)
        embedded = self.del_special_token(seq_list, embedded)  # !!!

        output = self.fc(embedded)
        output = self.fc1(output)
        output = self.fc1(output)
        output = self.fc1(output)
        output = self.linear_output_to_tag(output)

        output = output.reshape(-1, output.shape[2])
        tag_scores = F.log_softmax(output, dim=1)  # [batch_size*seq_len, tags_size]

        return tag_scores


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
