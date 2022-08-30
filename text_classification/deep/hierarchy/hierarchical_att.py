import argparse
import datetime
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from text_rnn_attention import RNNAttention


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
        # input: [batch_size, sent_len, seq_len]
        x = x.permute(1, 0, 2)  # [sent_len, batch_size, seq_len]
        output_list = []
        # word level
        for i in x:
            embed = self.embedding(i)  # [batch_size, seq_len, embed_dim]
            word_hidden, word_hn = self.model(embed)  # hidden: [batch_size, seq_len, hidden_dim * num_directions]
            # attention
            M = self.out_trans(word_hidden)  # [batch_size, seq_len, hidden_dim * num_directions]
            alpha = F.softmax(self.w(M), dim=1)
            word_hidden = torch.sum(word_hidden * alpha, 1)  # [batch_size, hidden_dim * num_directions]
            output_list.append(word_hidden)
        sent_in = torch.stack(output_list)  # [sent_len, batch_size, hidden_dim * num_directions]
        sent_in = sent_in.permute(1, 0, 2)  # [batch_size, sent_len, hidden_dim * num_directions]
        # sentence level
        sent_hidden, _ = self.model(sent_in)  # hidden: [batch_size, sent_len, hidden_dim * num_directions]
        M2 = self.out_trans(sent_hidden)
        alpha2 = F.softmax(self.w(M2), dim=1)
        sent_out = torch.sum(sent_hidden * alpha2, 1)  # [batch_size, hidden_dim * num_directions]
        out = F.relu(sent_out)
        out = self.fc_out(out)  # [batch_size, num_classes]
        return out

    def data_generator(self, input_path, output_path, batch_size, word_dict=None, shuffle=True):
        pad_size = self.pad_size

        print("Load input_data data from {}.".format(input_path))
        print("Load output_data from {}.".format(output_path))

        # 读取输入数据。
        # type(src) = list
        with open(input_path, 'r') as fp:
            input_data = fp.readlines()

        # 读取输出数据。
        # type(out) = list
        with open(output_path, 'r') as fp:
            output_data = fp.readlines()

        # 如果你的机器资源不够，一次跑不完数据，为了验证整个流程（pipeline）是对的，可以采用小一点的数据规模。
        # 比方说，这里设置了1000，你可以可以设置100或者200等。
        # 如果你可以跑全量数据，那么你可以注释掉这行代码。
        # 记得有一些可用可不用的代码，可以注释掉而不要直接删掉。免得以后用的时候还要重新写。
        # src = src[:1000]
        # out = out[:1000]

        # 是否将数据随机打乱？
        if shuffle:
            data = list(zip(input_data, output_data))
            random.shuffle(data)
            input_data, output_data = zip(*data)

        # 按照batch将数据输出给深度学习模型。
        # 注意，因为下面使用了yield而非return，所以这个函数是一个生成器。

        for i in range(0, len(input_data), batch_size):
            new_batch = []
            # 读取输入数据的下一个batch。
            for doc in input_data[i: i + batch_size]:
                new_doc = []
                # 将该文档的句子拆分开
                for sent in doc.strip().split('.'):
                    new_sent = []
                    # 将该句子的单词拆分开，并将字符的列表转化为id的列表。
                    for word in sent.strip().split():
                        new_sent.append(int(word_dict[word]) if word in word_dict else int(word_dict['UNK']))
                    # 对于seq_len的padding操作
                    while True:
                        if len(new_sent) > pad_size:
                            new_doc.append(torch.Tensor(new_sent[:pad_size]))
                            new_sent = new_sent[pad_size:-1]
                        elif len(new_sent) < pad_size:
                            lst = [0] * int(pad_size - len(new_sent))
                            new_sent.extend(lst)
                            new_doc.append(torch.Tensor(new_sent))
                            break
                        else:
                            new_doc.append(torch.Tensor(new_sent))
                            break
                # 将转化后的id列表append到batch_input中。
                # 因为要使用toch的pad函数，所以此处就要把每个id列表转化为torch的Tensor格式。
                new_batch.append(torch.stack(new_doc))
            batch_x = pad_sequence(new_batch, batch_first=True).detach().numpy()

            # 提取输出数据的下一个batch。
            batch_output = [int(label.strip()) for label in output_data[i: i+batch_size]]
            batch_y = np.array(batch_output)

            yield batch_x, batch_y


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
        print("The output is: {}".format(output_data))

        print("The test process is done.")
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Hierarchical Attention Model!')