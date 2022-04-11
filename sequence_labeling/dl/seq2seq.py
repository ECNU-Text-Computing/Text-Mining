#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Sequence to Sequence
======
A class for Seq2Seq.
参考资料：
《NLP FROM SCRATCH: TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION》
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#nlp-from-scratch-translation-with-a-sequence-to-sequence-network-and-attention
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.data_loader import DataLoader
from sequence_labeling.data_processor import DataProcessor
from sequence_labeling.utils.evaluate import Evaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        print('Initialized EncoderRNN.')

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, **config):
        super(DecoderRNN, self).__init__()
        self.config = config
        self.model_name = config['model_name']
        self.data_root = config['data_root']
        self.hidden_size = config['hidden_dim']
        self.output_size = config['tag_size']
        self.dropout_p = config['dropout_rate']
        self.epochs = config['num_epochs']
        self.learning_rate = config['learning_rate']

        vocab = DataProcessor(**self.config).load_vocab()
        self.vocab_size = len(vocab)
        tags = DataProcessor(**self.config).load_tags()
        self.SOS_token = tags['SOS']
        self.index_tag_dict = dict(zip(tags.values(), tags.keys()))

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        print('Initialized DecoderRNN.')

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def run_model(self, model, run_mode, data_path):
        encoder = EncoderRNN(self.vocab_size, self.hidden_size).to(device)
        decoder = model.to(device)
        encode_model_saved = self.data_root + self.config['encode_model_save_name']
        decode_model_saved = self.data_root + self.config['decode_model_save_name']

        if run_mode == 'train':
            print('Running {} model. Training...'.format(self.model_name))
            model.train()
            plot_losses = []
            loss_total = 0
            loss_average = 0
            data_counter = 0

            encoder_optimizer = optim.SGD(encoder.parameters(), lr=self.learning_rate)
            decoder_optimizer = optim.SGD(decoder.parameters(), lr=self.learning_rate)

            criterion = nn.NLLLoss()

            for iter in range(1, self.epochs + 1):
                batch_num = 1
                for x, x_len, y, y_len in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                                   run_mode=run_mode):
                    for i in range(len(x)):
                        data_counter += 1
                        input_tensor = torch.tensor(x[i]).long().view(-1, 1)  # [seq_len ,1]
                        target_tensor = torch.tensor(y[i]).long().view(-1, 1)

                        loss = self.model_train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
                                                decoder_optimizer, criterion)

                        loss_total += loss
                    loss_average = loss_total / data_counter
                    plot_losses.append(loss_average)

                    batch_num += 1

                print("Done Epoch {}. Proceeded {} batch data totally. Loss_average = {}".format(iter, batch_num,
                                                                                                 loss_average))

            n_batch = np.arange(1, len(plot_losses) + 1, 1)
            loss_list = np.array(plot_losses)
            plt.plot(n_batch, loss_list)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid()
            plt.show()

            torch.save(encoder, '{}'.format(encode_model_saved))
            torch.save(decoder, '{}'.format(decode_model_saved))

        elif run_mode == 'eval' or 'test':
            print('Running {} model. {}ing...'.format(self.model_name, run_mode))
            encoder = torch.load(encode_model_saved)
            decoder = torch.load(decode_model_saved)
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                for x, x_len, y, y_len in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                                   run_mode=run_mode):
                    y_predict = []
                    for i in range(len(x)):
                        input_tensor = torch.tensor(x[i]).long().view(-1, 1)  # [seq_len ,1]
                        input_length = input_tensor.size()[0]
                        encoder_hidden = encoder.initHidden()

                        for ei in range(input_length):
                            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                                     encoder_hidden)

                        decoder_input = torch.tensor([[self.SOS_token]], device=device)  # SOS
                        decoder_hidden = encoder_hidden

                        for di in range(input_length):
                            decoder_output, decoder_hidden = decoder(
                                decoder_input, decoder_hidden)

                            topv, topi = decoder_output.data.topk(1)
                            decoder_input = topi.squeeze().detach()
                            y_predict.append(self.index_tag_dict[topi.item()])

                    print("y_predict = {}".format(y_predict))

                    y = y.flatten()
                    y_true = []
                    for i in range(len(y)):
                        y_true.append(self.index_tag_dict[y[i]])
                    print("y_true = {}".format(y_true))

                    # 输出评价结果
                    print(Evaluator().classifyreport(y_true, y_predict))

        else:
            print("run_mode参数未赋值(train/eval/test)")

    def model_train(self, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    criterion):
        encoder_hidden = encoder.initHidden()  # [1, 1, hidden_size]/[num_layers * num_directions, batch_size, hidden_size]
        # print('***train(), 开始decode***\nencoder_hidden的shape: {}'.format(encoder_hidden.shape))

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)  # seq_len
        target_length = target_tensor.size(0)
        # print(target_tensor)

        loss = 0

        for ei in range(input_length):  # 依序处理seq
            # torch.nn.GRU输出：out和 ht
            # out的输出维度：[seq_len,batch_size,output_dim] / [1, 1, hidden_size]
            # ht的维度：[num_layers * num_directions, batch_size, hidden_size] / [1, 1, hidden_size]
            # out[-1]=ht[-1]
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            # print("Encoder处理seq中一个符号的结果。encoder_output的shape为: {}， encoder_hidden 的shape为: {}".format(encoder_output.shape, encoder_hidden.shape))

        # print("***train(), 开始decode***\n输入为decoder_input, decoder_hidden, encoder_outputs")
        # decoder_input的初始值为输出序列中的‘SOS_token’
        # decoder_hidden的初始值为encoder最后时间步的隐藏层编码
        decoder_input = torch.tensor([[self.SOS_token]], device=device)  # [1, 1]
        # print('decode_input的shape: {}'.format(decoder_input.shape))
        decoder_hidden = encoder_hidden  # [1, 1, hidden_size] / [num_layers * num_directions, batch_size, hidden_size]

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # Without teacher forcing: use its own predictions as the next input
            # topk():沿给定dim维度返回输入张量input中 k 个最大值。
            # 如果不指定dim，则默认为input的最后一维。
            # 返回一个元组 (values,indices)，其中indices是原始输入张量input中测元素下标
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length


if __name__ == '__main__':
    # 以run方式运行，需要将配置文件中"data_root"的值修改为"../Datasets/cmed/"。

    config_path = '../config/cmed/dl/cmed.dl.attndecoderrnn.norm.json'
    if not os.path.exists(config_path):
        raise RuntimeError("There is no {} config.".format(config_path))
    config = json.load(open(config_path, 'r'))
    print('config: ', config)

    data_path = config['data_root']
    model = DecoderRNN(**config)
    model.run_model(model, run_mode='train', data_path=data_path)
    model.run_model(model, run_mode='test', data_path=data_path)
