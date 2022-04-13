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

import os
import sys

import matplotlib.pyplot as plt
import json
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
        self.hidden_dim = hidden_size
        self.output_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(self.hidden_dim, self.output_size)
        print('Initialized EncoderRNN.')

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, **config):
        super(AttnDecoderRNN, self).__init__()
        self.config = config
        self.data_root = config['data_root']
        self.model_name = config['model_name']
        self.hidden_size = config['hidden_dim']
        self.dropout_p = config['dropout_rate']
        self.max_length = config['max_length']
        self.epochs = config['num_epochs']
        self.learning_rate = config['learning_rate']

        vocab = DataProcessor(**self.config).load_vocab()
        self.vocab_size = len(vocab)
        tags = DataProcessor(**self.config).load_tags()
        self.output_size = len(tags)
        self.SOS_token = tags['SOS']
        self.index_tag_dict = dict(zip(tags.values(), tags.keys()))

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        print('Initialized AttnDecoderRNN.')

    def forward(self, input, hidden, encoder_outputs):
        # input的shape: torch.Size([1, 1]) / decoder前一时间步的输出，初始化值为‘SOS_token’
        # hidden的shape: torch.Size([1, 1, 7]) / [1, 1, hidden_size]/初始值为encoder最后时间步的隐藏层编码
        # encoder_outputs的shape: torch.Size([20, 7]) / [MAX_LENGTH, hidden_size] / encoder的输出
        # print("****调用AttnDecoderRNN.forward()。\ninput的shape: {}".format(input.shape))
        # print("hidden的shape: {}".format(hidden.shape))
        # print("encoder_output的shape: {}\n".format(encoder_outputs.shape))
        embedded = self.embedding(input).view(1, 1, -1)  # [1, 1, 7]
        # print("input执行embedding后的shape: {}".format(embedded.shape))
        embedded = self.dropout(embedded)

        # 拼接embedded imput和hidden, 计算attention权重，归一化
        # decoder的每一个时间步都会计算一次，因而会针对性的产生attention权重
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)  # [1, 20] / 按最长seq计算权重分配

        # print(attn_weights.unsqueeze(0).shape)
        # print(encoder_outputs.unsqueeze(0).shape)

        # torch.bmm()传入的两个三维tensor，第一维必须相等，且第一个数组的第三维和第二个数组的第二维度要一致。
        # 此处两者分别为[1, 1, 20]、[1, 20, 7]
        # 实际效果是对encoder每一个时间步的输出赋予不同权重
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))  # [1, 1, 7]

        output = torch.cat((embedded[0], attn_applied[0]), 1)  # [1, 14]
        output = self.attn_combine(output).unsqueeze(0)  # [1, 1, 7]

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def run_model(self, model, run_mode, data_path):
        encoder = EncoderRNN(self.vocab_size, self.hidden_size).to(device)
        decoder = model.to(device)
        encode_model_saved = '{}{}_{}_encode_model.ckpt'.format(self.data_root, type(encoder).__name__,
                                                                type(decoder).__name__)
        decode_model_saved = '{}{}_{}_decode_model.ckpt'.format(self.data_root, type(encoder).__name__,
                                                                type(decoder).__name__)

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
                        encoder_hidden = encoder.init_hidden()

                        encoder_outputs = torch.zeros(self.max_length, encoder.hidden_dim, device=device)

                        for ei in range(input_length):
                            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                                     encoder_hidden)
                            encoder_outputs[ei] += encoder_output[0, 0]

                        decoder_input = torch.tensor([[self.SOS_token]], device=device)  # SOS
                        decoder_hidden = encoder_hidden
                        decoder_attentions = torch.zeros(self.max_length, self.max_length)
                        # decoded_words = []

                        for di in range(input_length):
                            decoder_output, decoder_hidden, decoder_attention = decoder(
                                decoder_input, decoder_hidden, encoder_outputs)
                            decoder_attentions[di] = decoder_attention.data
                            topv, topi = decoder_output.data.topk(1)
                            decoder_input = topi.squeeze().detach()
                            # decoded_words.append(index_tag_dict[topi.item()])
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
        encoder_hidden = encoder.init_hidden()  # [1, 1, hidden_size]/[num_layers * num_directions, batch_size, hidden_size]
        # print('***train(), 开始decode***\nencoder_hidden的shape: {}'.format(encoder_hidden.shape))

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)  # seq_len
        target_length = target_tensor.size(0)
        # print(target_tensor)

        encoder_outputs = torch.zeros(self.config['max_length'], encoder.hidden_dim, device=device)
        # print("按最大seq_len（20）定义encode_output，shape: {}".format(encoder_outputs.shape))

        loss = 0

        for ei in range(input_length):  # 依序处理seq
            # torch.nn.GRU输出：out和 ht
            # out的输出维度：[seq_len,batch_size,output_dim] / [1, 1, hidden_size]
            # ht的维度：[num_layers * num_directions, batch_size, hidden_size] / [1, 1, hidden_size]
            # out[-1]=ht[-1]
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            # print("train(): Encoder处理seq中一个符号的结果。encoder_output的shape为: {}， encoder_hidden 的shape为: {}".format(encoder_output.shape, encoder_hidden.shape))
            encoder_outputs[ei] = encoder_output[0, 0]  # encoder_output[0, 0]是一维张量 / [hidden_size]

        # print("Encoder处理seq序列的结果。encoder_outputs的shape为: {}\n".format(encoder_outputs.shape))  # [MAX_LENGTH, hidden_size]

        # print("***train(), 开始decode***\n输入为decoder_input, decoder_hidden, encoder_outputs")
        # decoder_input的初始值为输出序列中的‘SOS_token’
        # decoder_hidden的初始值为encoder最后时间步的隐藏层编码
        decoder_input = torch.tensor([[self.SOS_token]], device=device)  # [1, 1]
        # print('decode_input的shape: {}'.format(decoder_input.shape))
        decoder_hidden = encoder_hidden  # [1, 1, hidden_size] / [num_layers * num_directions, batch_size, hidden_size]

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # Without teacher forcing: use its own predictions as the next input
            # topk():沿给定dim维度返回输入张量input中 k 个最大值。
            # 如果不指定dim，则默认为input的最后一维。
            # 返回一个元组 (values,indices)，其中indices是原始输入张量input中测元素下标
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            # if decoder_input.item() == EOS_token:
            #     break

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

    data_path = "../Datasets/cmed/"
    model = AttnDecoderRNN(**config)
    model.run_model(model, run_mode='train', data_path=data_path)
    model.run_model(model, run_mode='test', data_path=data_path)
