#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: Transformer
======
A class for Transformer.
配置文件：cmed.dl.transformer.norm.json
"""
import math
import sys

import torch
import torch.nn as nn

sys.path.insert(0, '')
sys.path.insert(0, '..')
from s2s import SeqToSeq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class PositionalEncoding(nn.Module):
    def __init__(self, **config):
        super(PositionalEncoding, self).__init__()
        d_model = config['d_model']
        max_len = config['max_len']
        dropout_p = config['dropout_rate']
        self.dropout = nn.Dropout(p=dropout_p)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(SeqToSeq):
    def __init__(self, **config):
        super().__init__(**config)
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        self.d_ff = config['FeedForward_dim']

        self.src_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.padding_idx)
        self.tgt_emb = nn.Embedding(self.tags_size, self.d_model, padding_idx=self.padding_idx_tags)
        self.pos_emb = PositionalEncoding(**config)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff, dropout=self.dropout_p)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.n_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff, dropout=self.dropout_p)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=self.n_layers)

        self.projection = nn.Linear(self.d_model, self.tags_size, bias=False).to(device)

    def forward(self, src, src_lengths, trg):
        """Transformer的输入：两个LongTensor
        src: [batch_size, src_len]
        trg: [batch_size, tgt_len]
        """
        # mask
        # tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(device)
        # src_key_padding_mask = enc_inputs.data.eq(0).to(device)  # [N,S]
        # tgt_key_padding_mask = dec_inputs.data.eq(0).to(device)  # [N,T]
        # memory_key_padding_mask = src_key_padding_mask  # [N,S]

        enc_inputs = self.src_emb(src)
        enc_inputs = self.pos_emb(enc_inputs.transpose(0, 1)).to(device)
        enc_outputs = self.transformer_encoder(src=enc_inputs)  # [src_len, batch_size, d_model]

        # 解码
        batch_size, src_len = src.shape[0], src.shape[1]
        trg_tensor = trg.transpose(0, 1)  # [trg_len, batch_size]
        dec_outputs = torch.zeros(src_len, batch_size, self.tags_size)  # 保存所有序列解码所得output
        # 构建decoder的初始输入
        dec_input = torch.zeros(batch_size, 1)  # [batch_size, 1]
        dec_input[:, 0] = self.SOS_token
        dec_input = dec_input.long()
        # 依序解码
        for t in range(0, src_len):
            dec_input = self.tgt_emb(dec_input)  # [batch_size, 1, d_model]
            dec_input = dec_input.transpose(0, 1)  # [1, batch_size, d_model]
            dec_input = self.pos_emb(dec_input).to(device)  # [1, batch_size, d_model]
            dec_output = self.transformer_decoder(tgt=dec_input, memory=enc_outputs)  # [1, batch_size, d_model]
            # 更新dec_input
            dec_input = trg_tensor[t].unsqueeze(1)
            # 维度变换
            tag_scores = self.projection(dec_output.squeeze(0))  # [batch_size, tags_size]
            tag_scores = nn.functional.log_softmax(tag_scores, dim=1)  # [batch_size, tags_size]
            dec_outputs[t] = tag_scores

        dec_outputs = dec_outputs.transpose(0, 1)  # [batch_size, seq_len, tags_size]
        output = dec_outputs.reshape(-1, dec_outputs.shape[-1])  # [batch_size*seq_len, tags_size]

        return output


if __name__ == '__main__':
    print(sys.modules)
