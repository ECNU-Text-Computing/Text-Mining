#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
配置文件: cmed.dl.s2s.norm.json
Class: SeqToSeq
    初始化参数: **config
    forward()：
        输入：src: 2维列表，[batch_size, seq_len]
            trg: 2维列表.[batch_size, seq_len]
        输出：loss: 1个batch的loss，值为tensor,用item()转化为标量值
            dec_outputs: [seq_len, batch_size, self.tags_size], 保存decoder所有时间步的output
    eval_process(): 每训练1个epoch，评价模型效率
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '../sequence_labeling/dl')
sys.path.insert(0, '../sequence_labeling')
from sequence_labeling.data_loader import DataLoader
from sequence_labeling.data_processor import DataProcessor
from sequence_labeling.utils.evaluate import Evaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class SeqToSeq(nn.Module):
    def __init__(self, **config):
        super(SeqToSeq, self).__init__()
        self.config = config
        self.model_name = config['model_name']
        self.data_root = config['data_root']
        self.model_save_path = config['model_save_name']
        self.num_epochs = config['num_epochs']
        self.pad_token = config['pad_token']

        self.enc_embedding_dim = config['enc_embedding_dim']
        self.enc_hidden_dim = config['enc_hidden_dim']
        self.enc_bidirectional = True if config['enc_bidirectional'] == 'True' else False
        self.enc_n_directions = 2 if self.enc_bidirectional else 1
        self.enc_layers = config['enc_layers']
        self.enc_dropout_p = config['enc_dropout_rate']
        self.enc_learning_rate = config['enc_learning_rate']

        self.dec_embedding_dim = config['dec_embedding_dim']
        self.dec_hidden_dim = config['dec_hidden_dim']
        self.dec_bidirectional = True if config['dec_bidirectional'] == 'True' else False
        self.dec_n_directions = 2 if self.dec_bidirectional else 1
        self.dec_layers = config['dec_layers']
        self.dec_dropout_p = config['dec_dropout_rate']
        self.dec_learning_rate = config['dec_learning_rate']

        vocab = DataProcessor(**config).load_vocab()
        self.vocab_size = len(vocab)
        self.padding_idx = vocab[self.pad_token]
        self.tag_index_dict = DataProcessor(**self.config).load_tags()
        self.tags_size = len(self.tag_index_dict)
        self.SOS_token = self.tag_index_dict['SOS']
        self.EOS_token = self.tag_index_dict['EOS']
        self.index_tag_dict = dict(zip(self.tag_index_dict.values(), self.tag_index_dict.keys()))

        self.enc_embedding = nn.Embedding(self.vocab_size, self.enc_embedding_dim)
        self.enc_gru = nn.GRU(input_size=self.enc_embedding_dim, hidden_size=self.enc_hidden_dim,
                              num_layers=self.enc_layers, bidirectional=self.enc_bidirectional)
        self.enc_hidden_to_dec_hidden = nn.Linear(self.enc_hidden_dim * self.enc_n_directions, self.dec_hidden_dim)

        self.dec_embedding = nn.Embedding(self.tags_size, self.dec_embedding_dim)
        self.dec_gru = nn.GRU(input_size=self.dec_embedding_dim, hidden_size=self.dec_hidden_dim,
                              num_layers=self.dec_layers, bidirectional=self.dec_bidirectional)
        self.dec_output_to_tags = nn.Linear(self.dec_hidden_dim * self.dec_n_directions, self.tags_size)

        self.criterion_dict = {
            'NLLLoss': torch.nn.NLLLoss,
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss
        }
        self.criterion = self.criterion_dict[config['criterion_name']]()

        self.optimizer_dict = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam
        }
        self.optimizer = self.optimizer_dict[config['optimizer_name']]

        print('完成类{}的初始化'.format(self.__class__.__name__))

    def init_hidden_enc(self, batch_size):
        return torch.zeros(self.enc_layers * self.enc_n_directions, batch_size, self.enc_hidden_dim, device=device)

    def forward(self, src, trg):
        loss = 0
        src_tensor = torch.LongTensor(src)
        trg_tensor = torch.LongTensor(trg).transpose(0, 1)
        batch_size, seq_len = src_tensor.size()

        enc_init_hidden = self.init_hidden_enc(batch_size)
        enc_embedded = self.enc_embedding(src_tensor).transpose(0, 1)  # [seq_len, batch-size, enc_embedding_dim]
        enc_output, enc_hidden = self.enc_gru(enc_embedded, enc_init_hidden)
        # 若为双向
        if self.enc_bidirectional:
            enc_hidden = self.enc_hidden_to_dec_hidden(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=1))
        # 若为单向，但enc_hidden_dim != dec_hidden_dim
        else:
            enc_hidden = self.enc_hidden_to_dec_hidden(enc_hidden[-1, :, :])

        dec_outputs = torch.zeros(seq_len, batch_size, self.tags_size)  # 保存解码所有时间步的output
        dec_hidden = enc_hidden.unsqueeze(0).repeat(self.dec_layers * self.dec_n_directions, 1, 1)
        dec_input = torch.tensor([[self.SOS_token]], device=device).repeat(batch_size, 1)  # [batch_size, 1]
        for t in range(0, seq_len):
            dec_input = self.dec_embedding(dec_input).transpose(0, 1)  # [1, batch_size, dec_embedding_dim]
            dec_output, dec_hidden = self.dec_gru(dec_input, dec_hidden)
            dec_output = self.dec_output_to_tags(dec_output.view(-1, dec_output.shape[-1]))  # [batch_size, tags_size]
            dec_output = nn.functional.log_softmax(dec_output, dim=1)
            dec_outputs[t] = dec_output
            top1 = dec_output.argmax(1)
            dec_input = top1.unsqueeze(1).detach()

            loss += self.criterion(dec_output, trg_tensor[t])

        return loss, dec_outputs

    def run_train(self, model):
        print('Running {} model. Training...'.format(self.model_name))
        run_mode = 'train'
        model_saved = '{}{}_decoder.ckpt'.format(self.data_root, self.model_name)

        model.train()
        plot_losses = []
        best_valid_loss = float('inf')
        model_optimizer = self.optimizer(model.parameters(), lr=self.dec_learning_rate)

        for epoch in range(self.num_epochs):
            sum_loss = 0
            counter = 0
            for x, x_len, y, y_len in DataLoader(**self.config).data_generator(data_path=self.data_root,
                                                                               run_mode=run_mode):
                loss, _ = model(x, y)
                model_optimizer.zero_grad()
                loss.backward()
                model_optimizer.step()

                sum_loss += loss
                counter += len(x)

            average_loss = sum_loss / counter
            plot_losses.append(average_loss.item())  # loss是tensor，用item()取出值

            # 模型评价
            eval_loss, f1_score = self.evaluate(model)
            if eval_loss < best_valid_loss:
                best_valid_loss = eval_loss
                torch.save(model, '{}'.format(model_saved))
            print("Done Epoch {}. train_average_loss = {}".format(epoch, average_loss))
        self.pltshow(plot_losses)

    def evaluate(self, model):
        # print('Running {} model. Evaluating...'.format(self.model_name))
        run_mode = 'eval'
        average_loss, f1 = self.eval_process(model, run_mode)

        return average_loss, f1

    def test(self):
        print('Running {} model. Testing...'.format(self.model_name))
        run_mode = 'test'
        model_saved = '{}{}_decoder.ckpt'.format(self.data_root, self.model_name)
        model = torch.load(model_saved)

        average_loss, f1 = self.eval_process(model, run_mode)

    def eval_process(self, model, run_mode):
        model.eval()
        with torch.no_grad():
            sum_loss = 0
            counter = 0
            batch_num = 0
            for x, x_len, y, y_len in DataLoader(**self.config).data_generator(data_path=self.data_root,
                                                                               run_mode=run_mode):
                loss, dec_outputs = model(x, y)

                # 累加batch的loss，返回值为每条数据的平均loss（sum_loss / counter）
                sum_loss += loss
                counter += len(x)

                # 计算评价指标
                tag_scores = dec_outputs.transpose(0, 1).contiguous()  # [batch_size, seq_len, self.tags_size]
                predict_labels = []
                for i in range(tag_scores.shape[0]):
                    item = tag_scores[i]
                    predict = item.argmax(1)
                    pre_list = list(predict.numpy())
                    predict_labels.append(pre_list)
                predict_labels = torch.IntTensor(predict_labels).view(-1)
                predict_labels = predict_labels.numpy().tolist()
                predict_labels = self.index_to_tag(predict_labels)

                target_labels = torch.IntTensor(y).view(-1)
                target_labels = target_labels.numpy().tolist()
                target_labels = self.index_to_tag(target_labels)

                f1_score = Evaluator().f1score(target_labels, predict_labels)
                batch_num += 1
                print('第{}批测试数据， f1_score = {}'.format(batch_num, f1_score))

                # print(Evaluator().classifyreport(target_labels, predict_labels))

        return sum_loss / counter, f1_score

    def index_to_tag(self, y):
        y_tagseq = []
        for i in range(len(y)):
            y_tagseq.append(self.index_tag_dict[y[i]])
        return y_tagseq

    def pltshow(self, y_axis_values):
        x_axis_values = np.arange(1, len(y_axis_values) + 1, 1)
        y_axis_values = np.array(y_axis_values)
        plt.plot(x_axis_values, y_axis_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.show()
