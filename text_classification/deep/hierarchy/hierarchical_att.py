import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep.text_rnn_attention import RNNAttention


class HierAttNet(RNNAttention):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(HierAttNet, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                         dropout_rate, learning_rate, num_epochs, batch_size,
                                         criterion_name, optimizer_name, gpu, **kwargs)
        # 设置句子填充大小
        self.pad_size = 20
        if 'pad_size' in kwargs:
            self.pad_size = kwargs['pad_size']

        # 设置输出时通过的全连接层
        self.fc_out = nn.Linear(self.hidden_out, self.num_classes)

    def forward(self, x):
        # input_data: [batch_size, sent_len, seq_len]
        batch_size, sent_len, seq_len = x.size()
        x = x.reshape(-1, seq_len)  # [batch_size * sent_len, seq_len]
        embed_word = self.embedding(x)
        # word level
        word_hidden, word_hn = self.model(embed_word)  # hidden: [batch_size * sent_len, seq_len, hidden_dim * num_directions]
        M = self.out_trans(word_hidden)  # [batch_size * sent_len, seq_len, hidden_dim * num_directions]
        alpha = F.softmax(self.w(M), dim=1)
        word_hidden = torch.sum(word_hidden * alpha, 1)  # [batch_size * sent_len, hidden_dim * num_directions]
        # sentence level
        sent_in = word_hidden.reshape(batch_size, sent_len, -1)  # [batch_size, sent_len, hidden_dim * num_directions]
        sent_hidden, _ = self.model(sent_in)  # hidden: [batch_size, sent_len, hidden_dim * num_directions]
        M2 = self.out_trans(sent_hidden)
        alpha2 = F.softmax(self.w(M2), dim=1)
        sent_out = torch.sum(sent_hidden * alpha2, 1)  # [batch_size, hidden_dim * num_directions]
        out = F.relu(sent_out)
        out = self.fc_out(out)  # [batch_size, num_classes]
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

        input_data = torch.LongTensor([[[1, 2, 3, 4, 5],
                                        [2, 3, 4, 5, 6],
                                        [3, 4, 5, 6, 7]],
                                       [[1, 3, 5, 7, 9],
                                        [2, 4, 6, 8, 10],
                                        [1, 4, 8, 3, 6]]])  # [batch_size, sent_len, seq_len] = [2, 3, 5]
        output_data = model(input_data)
        print("The output_data is: {}".format(output_data))

        print("The test process is done.")
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Hierarchical Attention Model!')
