#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Sequence to Sequence
======
A class for seq2seq.
Input data on batch.
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
from .s2s_encoder_gru_b import EncoderGRU_B

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class DecoderGRU_B(nn.Module):
    def __init__(self, **config):
        super(DecoderGRU_B, self).__init__()
        self.config = config
        self.model_name = config['model_name']
        self.data_root = config['data_root']
        self.batch_size = config['batch_size']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.bidirectional = True if config['bidirectional'] == 'True' else False
        self.n_directions = 2 if self.bidirectional else 1  # 双向循环，输出的hidden是正向和反向hidden的拼接，所以要 *2
        self.dropout = config['dropout_rate']
        self.learning_rate = config['learning_rate']
        self.epochs = config['num_epochs']

        vocab = DataProcessor(**config).load_vocab()
        self.vocab_size = len(vocab)
        tags = DataProcessor(**config).load_tags()
        self.tags_size = len(tags)
        self.SOS_token = tags['SOS']
        self.EOS_token = tags['EOS']
        self.index_tag_dict = dict(zip(tags.values(), tags.keys()))

        self.criterion_dict = {
            'NLLLoss': torch.nn.NLLLoss,
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss
        }
        self.criterion = self.criterion_dict[config['criterion_name']]()

        self.output_size = self.tags_size
        self.embedding_dim = self.hidden_dim
        self.embedding = nn.Embedding(self.output_size, self.embedding_dim)
        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, bidirectional=self.bidirectional)  # GRU循环神经网络
        self.out = nn.Linear(self.hidden_dim * self.n_directions, self.output_size)  # 全连接层

        print('Initialized Model: {}.'.format(self.model_name))

    def forward(self, input, hidden):
        batch_size = input.size(0)
        output = self.embedding(input).view(-1, batch_size, self.embedding_dim)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.num_layers * self.n_directions, self.batch_size, self.hidden_dim, device=device)

    def run_model(self, model, run_mode, data_path):
        encoder = EncoderGRU_B().to(device)
        decoder = model.to(device)
        encode_model_saved = '{}{}_{}_encode_model.ckpt'.format(self.data_root, type(encoder).__name__,
                                                                type(decoder).__name__)
        decode_model_saved = '{}{}_{}_decode_model.ckpt'.format(self.data_root, type(encoder).__name__,
                                                                type(decoder).__name__)

        if run_mode == 'train':
            print('Running {} model. {}ing...'.format(self.model_name, run_mode))
            model.train()
            plot_losses = []
            loss_total = 0
            loss_average = 0
            data_counter = 0

            encoder_optimizer = optim.SGD(encoder.parameters(), lr=self.learning_rate)
            decoder_optimizer = optim.SGD(decoder.parameters(), lr=self.learning_rate)

            for iter in range(1, self.epochs + 1):
                for x, x_len, y, y_len in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                                   run_mode=run_mode):
                    data_counter += len(x)
                    input_tensor = torch.tensor(x).long()
                    target_tensor = torch.tensor(y).long()

                    loss = self.model_train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
                                            decoder_optimizer, self.criterion)
                    loss_total += loss

                    loss_average = loss_total / data_counter
                    plot_losses.append(loss_average)

                print("Done Epoch {}. Loss_average = {}".format(iter, loss_average))

            torch.save(encoder, '{}'.format(encode_model_saved))
            torch.save(decoder, '{}'.format(decode_model_saved))

            self.pltshow(plot_losses)

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

                    input_tensor = torch.tensor(x).long()

                    input_tensor = input_tensor.transpose(0, 1)  # [seq_len, batch_size]
                    input_length = input_tensor.size(0)  # seq_len
                    input_batch_size = input_tensor.size(1)
                    # print(input_length)

                    encoder_hidden = encoder.init_hidden(input_batch_size)
                    for ei in range(input_length):
                        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

                    decoder_input = torch.tensor([[self.SOS_token]], device=device).repeat(input_batch_size,
                                                                                           1)
                    decoder_hidden = encoder_hidden
                    for di in range(input_length):  # target_length = input_length
                        decoder_output, decoder_hidden = decoder(
                            decoder_input, decoder_hidden)
                        topv, topi = decoder_output.data.topk(1)
                        predict = topi.transpose(0, 1).squeeze().detach()
                        decoder_input = topi.squeeze().detach()
                        y_predict.append(list(np.array(predict)))

                    y_predict = torch.IntTensor(y_predict).transpose(0, 1)
                    y_predict = list(np.array(y_predict).flatten())

                    y_eval = [self.index_tag_dict[ei.item()] for ei in y_predict]
                    print('y_pred = {}'.format(y_eval))

                    y_true = []
                    y = y.flatten()
                    for i in range(len(y)):
                        y_true.append(self.index_tag_dict[y[i]])
                    print("y_true = {}".format(y_true))

                    # 输出评价结果
                    print(Evaluator().classifyreport(y_true, y_eval))

        else:
            print("run_mode参数未赋值(train/eval/test)")

    def model_train(self, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    criterion):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_tensor = input_tensor.transpose(0, 1)  # [seq_len, batch_size]
        # print('decoder-input_tensor trans size: {}'.format(input_tensor.shape))
        input_length = input_tensor.size(0)  # seq_len
        input_batch_size = input_tensor.size(1)
        # print(input_length)

        target_tensor = target_tensor.transpose(0, 1)
        target_length = target_tensor.size(0)
        # print(target_length)
        loss = 0

        encoder_hidden = encoder.init_hidden(input_batch_size)
        for ei in range(input_length):  # 依序处理seq
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)  # [1, batch_size, hidden_dim]

        # decoder_input的初始值为输出序列中的‘SOS_token’
        # decoder_hidden的初始值为encoder最后时间步的隐藏层编码
        decoder_input = torch.tensor([[self.SOS_token]], device=device).repeat(input_batch_size, 1)  # [1, batch_size]
        decoder_hidden = encoder_hidden
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # Without teacher forcing: use its own predictions as the next input
            # topk():沿给定dim维度返回输入张量input中 k 个最大值。
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def pltshow(self, y_list):
        x_array = np.arange(1, len(y_list) + 1, 1)
        y_array = np.array(y_list)
        plt.plot(x_array, y_array)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    # 以run方式运行，需要将配置文件中"data_root"的值修改为"../Datasets/cmed/"。

    config_path = '../config/cmed/dl/cmed.dl.s2s_docoder_gru_attn_bmm.norm.json'
    if not os.path.exists(config_path):
        raise RuntimeError("There is no {} config.".format(config_path))
    config = json.load(open(config_path, 'r'))
    print('config: ', config)

    data_path = config['data_root']
    model = DecoderGRU_B(**config)
    model.run_model(model, run_mode='train', data_path=data_path)
    model.run_model(model, run_mode='test', data_path=data_path)
