import argparse
import datetime

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from deep.base_model import BaseModel


class RNNAttention(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(RNNAttention, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                           dropout_rate, learning_rate, num_epochs, batch_size,
                                           criterion_name, optimizer_name, gpu, **kwargs)
        self.model_name = 'RNNAttention'
        # RNN模型参数设置
        # RNN堆叠的层数，默认为2层
        self.num_layers = 2
        if 'num_layers' in kwargs:
            self.num_layers = kwargs['num_layers']
        # RNN的方向性，双向则取值2；单向则取1
        self.num_directions = 2
        if 'num_directions' in kwargs:
            self.num_directions = kwargs['num_directions']
        self.bidirectional = True if self.num_directions == 2 else False
        # 使用的神经网络类型
        self.rnn_model = 'LSTM'
        if 'rnn_model' in kwargs:
            self.rnn_model = kwargs['rnn_model']

        # RNN模型初始化
        if self.rnn_model == 'LSTM':
            self.model_name = 'LSTMAttention'
            if self.bidirectional:
                self.model_name = 'BiLSTMAttention'
            self.model = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=self.num_layers,
                                 dropout=dropout_rate, batch_first=True, bidirectional=self.bidirectional)
        elif self.rnn_model == 'GRU':
            self.model_name = 'GRUAttention'
            if self.bidirectional:
                self.model_name = 'BiGRUAttention'
            self.model = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, num_layers=self.num_layers,
                                dropout=dropout_rate, batch_first=True, bidirectional=self.bidirectional)
        elif self.rnn_model == 'RNN':
            self.model_name = 'RNNAttention'
            if self.bidirectional:
                self.model_name = 'BiRNNAttention'
            self.model = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim, num_layers=self.num_layers,
                                dropout=dropout_rate, batch_first=True, bidirectional=self.bidirectional)
        else:
            print('No such RNN model!')

        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim // 2
        if 'hidden_dim2' in kwargs:
            self.hidden_dim2 = kwargs['hidden_dim2']

        self.tanh = nn.Tanh()
        if self.bidirectional:
            self.hidden_out = self.hidden_dim * 2
        else:
            self.hidden_out = self.hidden_dim
        self.out_trans = nn.Linear(self.hidden_out, self.hidden_out)
        self.w = nn.Linear(self.hidden_out, 1, bias=False)
        # self.attention = nn.Linear(self.hidden_dim * 2, 1, bias=False)  # 无偏差项单一输出神经网络作为注意力机制
        self.fc1 = nn.Linear(self.hidden_out, self.hidden_dim2)
        self.fc2 = nn.Linear(self.hidden_dim2, self.num_classes)

    def forward(self, x):
        # 词嵌入
        x_embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        print('x_embed的形状是：{}'.format(x_embed.size()))
        # 将句子填充到统一长度。注意pad_sequence的输入序列需是Tensor的tuple，因此pad操作后需对数据维度进行压缩
        # x_pad = pad_sequence([x_embed], batch_first=True).squeeze(0)  # [batch_size, seq_len, embed_dim]
        # print('x_pad的形状是：{}'.format(x_pad.size()))
        x_input = self.drop_out(x_embed)
        # 为高效地读取数据进行训练，需对填充好的数据进行压缩
        seq_len = [s.size(0) for s in x_input]
        packed_input = pack_padded_sequence(x_input, seq_len, batch_first=True, enforce_sorted=False)
        # 进行模型训练
        packed_output, ht = self.model(packed_input, None)  # [batch_size, seq_len, hidden_dim * num_directions]
        # 将压缩过的数据还原
        out_rnn, lens = pad_packed_sequence(packed_output, batch_first=True)

        # 计算attention
        M = self.tanh(self.out_trans(out_rnn))
        alpha = F.softmax(self.w(M), dim=1)  # 注意：进行softmax操作时一定要指定维度
        out = out_rnn * alpha  # [batch_size, seq_len, hidden_dim * num_directions]
        print('out的形状是：{}'.format(out.size()))
        out = torch.sum(out, dim=1)  # [batch_size, hidden_dim * num_directions]
        print('out2的形状是：{}'.format(out.size()))
        out = F.relu(out)
        # 通过两个全连接层后输出
        out = self.fc1(out)  # [batch_size, hidden_dim2]
        print('out3的形状是：{}'.format(out.size()))
        out = self.fc2(out)  # [batch_size, num_classes]
        return out


if __name__ == '__main__':
    if __name__ == '__main__':
        start_time = datetime.datetime.now()
        parser = argparse.ArgumentParser(description='Process some description.')
        parser.add_argument('--phase', default='test', help='the function name.')

        args = parser.parse_args()

        if args.phase == 'test':
            vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, \
            num_epochs, batch_size, criterion_name, optimizer_name, gpu \
                = 100, 256, 32, 2, 0.5, 0.0001, 3, 64, 'CrossEntropyLoss', 'Adam', 0

            model = RNNAttention(vocab_size, embed_dim, hidden_dim, num_classes,
                                 dropout_rate, learning_rate, num_epochs, batch_size,
                                 criterion_name, optimizer_name, gpu)

            input = torch.LongTensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
                                      [1, 3, 5, 7, 9], [2, 4, 6, 8, 10],
                                      [1, 4, 8, 3, 6]])  # [batch_size, seq_len] = [6, 5]
            output = model(input)
            print(output)

            print('The test process is done.')

        else:
            print("There is no {} function. Please check your command.".format(args.phase))
        end_time = datetime.datetime.now()
        print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

        print('Done RNNAttention!')
