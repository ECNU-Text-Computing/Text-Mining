import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..text_rnn import TextRNN


class HierAttNet(TextRNN):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(HierAttNet, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                         dropout_rate, learning_rate, num_epochs, batch_size,
                                         criterion_name, optimizer_name, gpu, **kwargs)
        self.out_trans = nn.Linear(self.embed_dim, self.embed_dim)
        self.w = nn.Linear(self.embed_dim, 1, bias=False)

    def forward(self, x):
        # input: [batch_size, seq_len]
        output_list = []
        for i in x:
            # RNN的输入应为3维，此处使用unsqueeze增加一维（batch_size为1）
            embed = self.embedding(i).unsqueeze(0)  # [1, seq_len, embed_dim]
            print(embed.size())
            word_hidden, word_hn = self.model(embed)  # hidden: [1, seq_len, hidden_dim * self.num_directions]
            M = self.out_trans(word_hidden)  # [1, seq_len, hidden_dim * self.num_directions]
            alpha = F.softmax(self.w(M), dim=1)
            word_out = word_hidden * alpha  # [1, seq_len, hidden_dim * self.num_directions]
            print(word_out.size())
            output_list.append(word_out)
        seq_input = torch.cat(output_list, 0)
        print(seq_input.size())
        seq_hidden, _ = self.model(seq_input)
        M2 = self.out_trans(seq_hidden)
        alpha2 = F.softmax(self.w(M2), dim=1)
        seq_out = seq_hidden * alpha2
        out = torch.sum(seq_out, dim=1)  # [batch_size, hidden_dim * self.num_directions]
        out = F.relu(out)
        # 通过全连接层后输出
        out = self.fc_out(out)
        return out

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
        vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu, num_layers, num_directions \
            = 100, 64, 32, 2, 0.5, 0.0001, 3, 8, 'CrossEntropyLoss', 'Adam', 0, 2, 2

        # 测试所用为2层双向LSTM
        model = HierAttNet(vocab_size, embed_dim, hidden_dim, num_classes,
                     dropout_rate, learning_rate, num_epochs, batch_size,
                     criterion_name, optimizer_name, gpu, num_layers=num_layers, num_directions=num_directions)

        input_data = torch.LongTensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
                                       [1, 3, 5, 7, 9], [2, 4, 6, 8, 10],
                                       [1, 4, 8, 3, 6]])  # [batch_size, seq_len] = [6, 5]
        output_data = model(input_data)
        print("The output is: {}".format(output_data))

        print("The test process is done.")
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Hierarchical Attention Model!')