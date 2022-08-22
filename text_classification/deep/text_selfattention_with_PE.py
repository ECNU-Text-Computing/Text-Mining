'''
Class SelfAttentionWithPE
======
Using nn.TransformerEncoderLayer() to construct self-attention and feedforward network
With positional encoding
'''

import datetime
import argparse
import math
import torch
import torch.nn as nn
from deep.base_model import BaseModel


class SelfAttentionWithPE(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(SelfAttentionWithPE, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                                  dropout_rate, learning_rate, num_epochs, batch_size,
                                                  criterion_name, optimizer_name, gpu, **kwargs)
        # 基本参数设置
        # encoder layer堆叠层数
        self.num_layers = 6
        if 'num_layers' in kwargs:
            self.num_layers = kwargs['num_layers']
        # 注意力头个数
        self.num_heads = 1
        if 'num_heads' in kwargs:
            self.num_heads = kwargs['num_heads']
        # 位置编码矩阵
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout_rate)
        # 包含self-attention和feed forward的encoder层
        encoder_layer = nn.TransformerEncoderLayer(self.embed_dim, self.num_heads, self.hidden_dim, self.dropout_rate,
                                                   device=self.device)
        # encoder层堆叠
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

    # 模型前向传播
    def forward(self, x):
        # src: [batch_size, seq_len]
        content = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        content = content * math.sqrt(self.embed_dim)  # [batch_size, seq_len, embed_dim]
        content = self.pos_encoder(content)  # [batch_size, seq_len, embed_dim]
        output = self.transformer_encoder(content)  # [batch_size, seq_len, embed_dim]
        output = torch.mean(output, dim=1)  # [batch_size, embed_dim]
        output = self.fc1(output)  # [batch_size, hidden_dim]
        output = self.fc_out(output)  # [batch_size, num_classes]
        return output


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description="Process some description.")
    parser.add_argument('--phase', default='test', help='the function name')
    args = parser.parse_args()

    if args.phase == 'test':
        print("This is a test process.")

        vocab_size, embed_dim, hidden_dim, num_classes, \
        dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu = 100, 64, 32, 2, 0.5, 0.001, 3, 32, 'CrossEntropyLoss', 'Adam', 0

        # 测试用例为单头、堆叠6层encoder的模型
        model = SelfAttentionWithPE(vocab_size, embed_dim, hidden_dim, num_classes,
                                    dropout_rate, learning_rate, num_epochs, batch_size,
                                    criterion_name, optimizer_name, gpu)

        src = torch.LongTensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 4, 2, 7, 5]])  # [batch_size, seq_len] = [3, 5]
        out = model(src)
        print("The output is: {}".format(out))

        print("The test process is done.")

    else:
        print("There is no {} function.".format(args.phase))

    end_time = datetime.datetime.now()
    print("{} takes {} seconds.".format(args.phase, (end_time - start_time).seconds))
    print("Done SelfAttentionWithPE.")
