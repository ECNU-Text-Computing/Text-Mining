#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: Bert_S2S_Attn
======
A class for S2S with dotproduct attention using Bert embedding.
配置文件：cmed.dl.bert_s2s_attn.norm.json
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_s2s import Bert_S2S

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class Bert_S2S_Attn(Bert_S2S):
    def __init__(self, **config):
        super().__init__(**config)

        self.attn_combine = nn.Linear(self.dec_embedding_dim + self.enc_hidden_dim * self.enc_n_directions,
                                      self.dec_embedding_dim)

    def forward(self, src, trg):
        batch_size, seq_len = trg.size()
        trg_tensor = trg.transpose(0, 1)

        # 编码
        batch = self.tokenizer(src, padding=True, truncation=True, return_tensors="pt")
        enc_embedded = self.bert_model(**batch).hidden_states[0]
        # 从embedded中删除表示[CLS],[SEP]的向量
        enc_embedded = self.del_special_token(src, enc_embedded).transpose(0, 1)

        enc_init_hidden = self.init_hidden_enc(batch_size)
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

            dec_input = trg_tensor[t].unsqueeze(1)
            # top1 = dec_output.argmax(1)
            # dec_input = top1.unsqueeze(1).detach()

            dec_output = self.dec_output_to_tags(
                dec_output.reshape(-1, dec_output.shape[-1]))  # [batch_size, tags_size]
            dec_output = nn.functional.log_softmax(dec_output, dim=1)
            dec_outputs[t] = dec_output

        dec_outputs = dec_outputs.transpose(0, 1)
        output = dec_outputs.reshape(-1, dec_outputs.shape[2])

        tag_scores = F.log_softmax(output, dim=1)  # [batch_size*seq_len, tags_size]

        return tag_scores


if __name__ == '__main__':
    print(sys.modules)
