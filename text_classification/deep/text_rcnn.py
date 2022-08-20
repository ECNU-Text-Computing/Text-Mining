# RCNN的卷积层指的是将该词的上下文信息和自身嵌入一起考虑而非使用卷积层，不存在滑动行为只是一个包括上下文信息的双向RNN，所以实现中没有卷积方法
# 词嵌入和线性层构成的简易模型
import argparse
import datetime
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from text_rnn import TextRNN
from torch import nn
import torch.nn.functional as F


class TextRCNN(TextRNN):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(TextRCNN, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)
        self.model_name = 'TextRCNN'
        self.pad_size = 64
        if 'pad_size' in kwargs:
            self.pad_size = kwargs['pad_size']

        self.maxpool = nn.MaxPool1d(self.pad_size)
        self.out_trans = nn.Linear(self.hidden_dim * self.num_directions + self.embed_dim,
                                   self.hidden_dim * self.num_directions)

    def forward(self, x):
        # input: [batch_size, seq_len]
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        # 将句子填充到统一长度。注意pad_sequence的输入序列需是Tensor的tuple，因此pad操作后需对数据维度进行压缩
        # x_pad = pad_sequence([x_embed], batch_first=True).squeeze(0)  # [batch_size, seq_len, embed_dim]
        # print('x_pad的形状是：{}'.format(x_pad.size()))
        input = self.drop_out(embed)
        seq_len = [s.size(0) for s in input]
        packed_input = pack_padded_sequence(input, seq_len, batch_first=True, enforce_sorted=False)
        packed_output, ht = self.model(packed_input, None)  # [batch_size, seq_len, hidden_dim * num_directions]
        out_rnn, _ = pad_packed_sequence(packed_output, total_length=self.pad_size, batch_first=True)  # size同上

        out = torch.cat((embed, out_rnn), 2)  # [batch_size, seq_len, hidden_dim * num_directions + embed_dim]
        out = self.out_trans(F.tanh(out))  # [batch_size, seq_len, hidden_dim * num_directions]
        out = out.permute(0, 2, 1)  # [batch_size, hidden_dim * num_directions, 1]
        out = self.maxpool(out).squeeze(-1)  # [batch_size, hidden_dim * num_directions]
        out = self.drop_out(out)
        out = self.fc_out(out)  # [batch_size, num_classes]
        return out



if __name__ == '__main__':
    if __name__ == '__main__':
        start_time = datetime.datetime.now()
        parser = argparse.ArgumentParser(description='Process some description.')
        parser.add_argument('--phase', default='test', help='the function name.')

        args = parser.parse_args()

        if args.phase == 'test':
            # For testing our model, we can set the hyper-parameters as follows.
            vocab_size, embed_dim, hidden_dim, num_classes, \
            num_layers, \
            dropout_rate, learning_rate, num_epochs, batch_size, \
            criterion_name, optimizer_name, gpu = \
                200, 64, 64, 2, 2, 0.5, 0.0001, 1, 64, 'CrossEntropyLoss', 'Adam', 1

            # new an objective.
            model = TextRCNN(vocab_size, embed_dim, hidden_dim, num_classes,
                             num_layers,
                             dropout_rate, learning_rate, num_epochs, batch_size,
                             criterion_name, optimizer_name, gpu)

            # a simple example of the input_data.
            input_data = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
            input_data = torch.LongTensor(input_data)  # input_data: [batch_size, seq_len] = [3, 5]

            # the designed model can produce an output.
            output_data = model(input_data)
            print(output_data)

            print('This is a test process.')
        else:
            print('error! No such method!')
        end_time = datetime.datetime.now()
        print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

        print('Done Text RCNN!')
