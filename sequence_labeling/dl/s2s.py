#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
配置文件: cmed.dl.s2s.norm.json
Class: SeqToSeq
    初始化参数: **config
    forward()：
        输入：src: 2维张量，[batch_size, seq_len]
            trg: 2维张量.[batch_size, seq_len]
        输出：output: [batch_size*seq_len, tags_size], 保存decoder所有时间步的输出
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '../sequence_labeling/dl')
sys.path.insert(0, '../sequence_labeling')
from sequence_labeling.dl.base_model import BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class SeqToSeq(BaseModel):
    def __init__(self, **config):
        super().__init__(**config)
        self.enc_embedding_dim = config['enc_embedding_dim']
        self.enc_hidden_dim = config['enc_hidden_dim']
        self.enc_bidirectional = True if config['enc_bidirectional'] == 'True' else False
        self.enc_n_directions = 2 if self.enc_bidirectional else 1
        self.enc_layers = config['enc_layers']
        self.enc_dropout_p = config['enc_dropout_rate']

        self.dec_embedding_dim = config['dec_embedding_dim']
        self.dec_hidden_dim = config['dec_hidden_dim']
        self.dec_bidirectional = True if config['dec_bidirectional'] == 'True' else False
        self.dec_n_directions = 2 if self.dec_bidirectional else 1
        self.dec_layers = config['dec_layers']
        self.dec_dropout_p = config['dec_dropout_rate']

        self.SOS_token = self.tag_index_dict['SOS']
        self.EOS_token = self.tag_index_dict['EOS']

        # 编码器设置
        self.enc_embedding = nn.Embedding(self.vocab_size, self.enc_embedding_dim)
        self.enc_gru = nn.GRU(input_size=self.enc_embedding_dim, hidden_size=self.enc_hidden_dim,
                              num_layers=self.enc_layers, bidirectional=self.enc_bidirectional)
        self.enc_hidden_to_dec_hidden = nn.Linear(self.enc_hidden_dim * self.enc_n_directions, self.dec_hidden_dim)

        # 解码器设置
        self.dec_embedding = nn.Embedding(self.tags_size, self.dec_embedding_dim)
        self.dec_gru = nn.GRU(input_size=self.dec_embedding_dim, hidden_size=self.dec_hidden_dim,
                              num_layers=self.dec_layers, bidirectional=self.dec_bidirectional)
        self.dec_output_to_tags = nn.Linear(self.dec_hidden_dim * self.dec_n_directions, self.tags_size)

    def init_hidden_enc(self, batch_size):
        return torch.zeros(self.enc_layers * self.enc_n_directions, batch_size, self.enc_hidden_dim, device=device)

    def forward(self, src, src_lengths, trg):
        src_tensor = src
        trg_tensor = trg.transpose(0, 1)
        batch_size, seq_len = src_tensor.size()

        # 编码
        enc_init_hidden = self.init_hidden_enc(batch_size)
        enc_embedded = self.enc_embedding(src_tensor).transpose(0, 1)  # [seq_len, batch-size, enc_embedding_dim]
        enc_output, enc_hidden = self.enc_gru(enc_embedded, enc_init_hidden)
        # 若为双向
        if self.enc_bidirectional:
            enc_hidden = self.enc_hidden_to_dec_hidden(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=1))
        # 若为单向，但enc_hidden_dim != dec_hidden_dim
        else:
            enc_hidden = self.enc_hidden_to_dec_hidden(enc_hidden[-1, :, :])

        # 解码
        dec_outputs = torch.zeros(seq_len, batch_size, self.tags_size)  # 保存解码所有时间步的output
        dec_hidden = enc_hidden.unsqueeze(0).repeat(self.dec_layers * self.dec_n_directions, 1, 1)
        dec_input = torch.tensor([[self.SOS_token]], device=device).repeat(batch_size, 1)  # [batch_size, 1]
        for t in range(0, seq_len):
            dec_input = self.dec_embedding(dec_input).transpose(0, 1)  # [1, batch_size, dec_embedding_dim]
            dec_output, dec_hidden = self.dec_gru(dec_input, dec_hidden)
            # 更新dec_input
            dec_input = trg_tensor[t].unsqueeze(1)
            # top1 = dec_output.argmax(1)
            # dec_input = top1.unsqueeze(1).detach()

            dec_output = self.dec_output_to_tags(
                dec_output.reshape(-1, dec_output.shape[-1]))  # [batch_size, tags_size]
            dec_output = nn.functional.log_softmax(dec_output, dim=1)
            dec_outputs[t] = dec_output

        dec_outputs = dec_outputs.transpose(0, 1)  # [batch_size, seq_len, tags_size]
        output = dec_outputs.reshape(-1, dec_outputs.shape[2])  # [batch_size*seq_len, tags_size]

        tag_scores = F.log_softmax(output, dim=1)  # [batch_size*seq_len, tags_size]

        return tag_scores


if __name__ == '__main__':
    print(sys.modules)
