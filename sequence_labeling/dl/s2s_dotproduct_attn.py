#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
配置文件: cmed.dl.s2s_dotproduct_attn.norm.json
Class: SeqToSeq_DotProductAttn
    实现DotProduct Attention，父类为s2s.py中的SeqToSeq.
    初始化参数: **config
    forward()：
        输入：src: 2维列表，[batch_size, seq_len]
            trg: 2维列表.[batch_size, seq_len]
        输出：loss: 1个batch的loss，值为tensor,用item()转化为标量值
            dec_outputs: [seq_len, batch_size, self.tags_size], 保存decoder所有时间步的output
"""

import sys

import torch
import torch.nn as nn

sys.path.insert(0, '../sequence_labeling/dl')
sys.path.insert(0, '../sequence_labeling')
from s2s import SeqToSeq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class SeqToSeq_DotProductAttn(SeqToSeq):
    def __init__(self, **config):
        super().__init__(**config)

        self.attn_combine = nn.Linear(self.dec_embedding_dim + self.enc_hidden_dim * self.enc_n_directions,
                                      self.dec_embedding_dim)

    def forward(self, src, trg):
        loss = 0
        src_tensor = torch.LongTensor(src)
        trg_tensor = torch.LongTensor(trg).transpose(0, 1)
        batch_size, seq_len = src_tensor.size()

        # 编码
        # enc_output: [seq_len, batch_size, enc_hidden_dim * enc_n_direction]
        # enc_hidden: [batch_size, hidden_dim]
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
        # 注意力权重映射
        attn = nn.Linear(self.dec_embedding_dim + self.dec_hidden_dim, seq_len)
        for t in range(0, seq_len):
            dec_input = self.dec_embedding(dec_input).transpose(0, 1)  # [1, batch_size, dec_embedding_dim]

            # DotProductAttention
            # 构建权重张量, [batch_size, seq_len]
            attn_weights = nn.functional.softmax(attn(torch.cat((dec_input[0], enc_hidden), 1)), dim=1)
            # 应用权重, [batch_size, 1, enc_hidden_dim * enc_n_direction]
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_output.transpose(0, 1))
            # 生成新的decoder输入, [1, batch_size, enc_embedding_dim]
            dec_input = self.attn_combine(torch.cat((dec_input[0], attn_applied.squeeze()), 1)).unsqueeze(0)
            dec_input = nn.functional.relu(dec_input)

            # decoder
            dec_output, dec_hidden = self.dec_gru(dec_input, dec_hidden)
            dec_output = self.dec_output_to_tags(dec_output.view(-1, dec_output.shape[-1]))  # [batch_size, tags_size]
            dec_output = nn.functional.log_softmax(dec_output, dim=1)
            dec_outputs[t] = dec_output
            top1 = dec_output.argmax(1)
            dec_input = top1.unsqueeze(1).detach()

            loss += self.criterion(dec_output, trg_tensor[t])

        return loss, dec_outputs


if __name__ == '__main__':
    print(sys.modules)
