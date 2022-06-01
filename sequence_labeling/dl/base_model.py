#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Base_Model
======
A class for BaseModel for all basic deep learning models.
配置文件：cmed.dl.base_model.norm.json
forward(X, X_lengths)：
    X：padding后的输入数据。2维列表，size = [batch_size, seq_len]
    X_lengths: 输入数据的真实长度。1维列表，size = [batch_size]
    返回值：tag_scores。2维张量，shape = [batch_size*seq_len, tags_size]
"""
import argparse
import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.data_loader import DataLoader
from sequence_labeling.data_processor import DataProcessor
from sequence_labeling.utils.evaluate import Evaluator
from sequence_labeling.utils.evaluate_2 import Metrics

torch.manual_seed(1)


class BaseModel(nn.Module):

    def __init__(self, **config):
        super(BaseModel, self).__init__()
        self.config = config
        self.model_name = config['model_name']
        self.data_root = config['data_root']
        self.model_save_path = config['model_save_name']

        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.layers = config['num_layers']
        self.bidirectional = True if config['bidirectional'] == 'True' else False
        self.n_directions = 2 if self.bidirectional else 1
        self.dropout_p = config['dropout_rate']
        self.pad_token = config['pad_token']

        self.num_epochs = config['num_epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']

        self.criterion_dict = {
            'NLLLoss': torch.nn.NLLLoss,
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
        }
        self.criterion_name = config['criterion_name']
        if self.criterion_name in self.criterion_dict:
            self.criterion = self.criterion_dict[self.criterion_name]()

        self.optimizer_dict = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam
        }
        self.optimizer_name = config['optimizer_name']
        if self.optimizer_name not in self.optimizer_dict:
            raise ValueError("There is no optimizer_name: {}.".format(self.optimizer_name))

        self.tag_index_dict = DataProcessor(**self.config).load_tags()
        self.tags_size = len(self.tag_index_dict)
        self.vocab = DataProcessor(**self.config).load_vocab()
        self.vocab_size = len(self.vocab)
        self.padding_idx = self.vocab[self.pad_token]
        self.padding_idx_tags = self.tag_index_dict[self.pad_token]

        self.word_embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim,
                                            padding_idx=self.padding_idx)

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.layers,
                            bidirectional=self.bidirectional, batch_first=True)

        # 将模型输出映射到标签空间
        self.output_to_tag = nn.Linear(self.hidden_dim * self.n_directions, self.tags_size)

        # print('完成类{}的初始化'.format(self.__class__.__name__))

    def forward(self, x, x_lengths, y):
        embeded = self.word_embeddings(x)
        embeded = rnn_utils.pack_padded_sequence(embeded, x_lengths, batch_first=True)
        output, _ = self.lstm(embeded)  # [batch_size, seq_len, hidden_dim]
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        out = output.reshape(-1, output.shape[2])  # 降维为[batch_size*seq_len, hidden_dim]
        out = self.output_to_tag(out)  # [batch_size*seq_len, tags_size]

        tag_scores = F.log_softmax(out, dim=1)  # [batch_size*seq_len, tags_size]

        return tag_scores

    def run_train(self, model, data_path):
        print('Running {} model. Training...'.format(self.model_name))
        run_mode = 'train'
        optimizer = self.optimizer_dict[self.optimizer_name](model.parameters(), lr=self.learning_rate)
        model.train()

        train_losses = []  # 记录每一batch的loss
        valid_losses = []
        valid_f1_scores = []
        best_valid_loss = float('inf')
        for epoch in range(self.num_epochs):
            train_loss = 0
            train_data_num = 0  # 统计数据量
            for seq_list, tag_list in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                               run_mode=run_mode):
                batch_input, batch_output = self.data_to_index(seq_list, tag_list)
                x, x_len, y, y_len = self.padding(batch_input, batch_output)
                train_data_num += len(x)
                batch_x = torch.tensor(x).long()
                batch_y = torch.tensor(y).long()

                tag_scores = model(batch_x, x_len, batch_y)

                batch_y = batch_y.view(-1)
                loss = self.criterion(tag_scores, batch_y)

                model.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_losses.append(train_loss / train_data_num)  # 记录评价结果

            # 模型验证
            avg_eval_loss, eval_f1 = self.evaluate(model, data_path)
            print("Done Epoch{}. Eval_Loss={}".format(epoch + 1, avg_eval_loss))
            valid_losses.append(avg_eval_loss)
            valid_f1_scores.append(eval_f1)

            # 保存参数最优的模型
            if avg_eval_loss < best_valid_loss:
                best_valid_loss = avg_eval_loss
                model_save_path = '{}{}_{}'.format(self.data_root, self.model_name, self.model_save_path)
                torch.save(model, '{}'.format(model_save_path))

        # self.prtplot(train_losses)
        # self.prtplot(valid_losses)

        return train_losses, valid_losses, valid_f1_scores

    def evaluate(self, model, data_path):
        # print('Running {} model. Evaluating...'.format(self.model_name))
        run_mode = 'eval'

        avg_loss, y_true, y_predict = self.eval_process(model, run_mode, data_path)
        f1_score = Evaluator().f1score(y_true, y_predict)

        return avg_loss, f1_score

    def test(self, data_path):
        print('Running {} model. Testing data in folder: {}'.format(self.model_name, data_path))
        run_mode = 'test'
        best_model_path = '{}{}_{}'.format(self.data_root, self.model_name, self.model_save_path)
        model = torch.load(best_model_path)

        avg_loss, y_true, y_predict = self.eval_process(model, run_mode, data_path)

        # 输出评价结果
        print(Evaluator().classifyreport(y_true, y_predict))
        f1_score = Evaluator().f1score(y_true, y_predict)

        # 输出混淆矩阵
        metrix = Metrics(y_true, y_predict)
        metrix.report_scores()
        metrix.report_confusion_matrix()

        return f1_score

    def eval_process(self, model, run_mode, data_path):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            data_num = 0
            for seq_list, tag_list in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                               run_mode=run_mode):
                batch_input, batch_output = self.data_to_index(seq_list, tag_list)
                x, x_len, y, y_len = self.padding(batch_input, batch_output)
                data_num += len(x)
                batch_x = torch.tensor(x).long()
                batch_y = torch.tensor(y).long()
                tag_scores = model(batch_x, x_len, batch_y)

                batch_y = batch_y.view(-1)
                loss = self.criterion(tag_scores, batch_y)
                total_loss += loss.item()

                # 返回每一行中最大值的索引
                predict = torch.max(tag_scores, dim=1)[1]  # 1维张量
                y_predict = list(predict.numpy())
                y_predict = self.index_to_tag(y_predict)
                # print(y_predict)

                y_true = y.flatten()
                y_true = self.index_to_tag(y_true)
                # print(y_true)

                # 专为测试评价结果增加代码
                # if run_mode == 'test':
                #     seq_len = len(x[0])
                #     for i in range(len(x)):
                #         print(self.index_to_vocab(x[i]))
                #         print(y_true[i*seq_len:(i+1)*seq_len-1])
                #         print(y_predict[i*seq_len:(i+1)*seq_len-1])

            return total_loss / data_num, y_true, y_predict

    def index_to_tag(self, y):
        index_tag_dict = dict(zip(self.tag_index_dict.values(), self.tag_index_dict.keys()))
        y_tagseq = []
        for i in range(len(y)):
            y_tagseq.append(index_tag_dict[y[i]])
        return y_tagseq

    def index_to_vocab(self, x):
        index_vocab_dict = dict(zip(self.vocab.values(), self.vocab.keys()))
        x_vocabseq = []
        for i in range(len(x)):
            x_vocabseq.append(index_vocab_dict[x[i]])
        return x_vocabseq

    def data_to_index(self, input, output):
        batch_input = []
        batch_output = []
        for line in input:  # 依据词典，将input转化为向量
            new_line = []
            for word in line.strip().split():
                new_line.append(int(self.vocab[word]) if word in self.vocab else int(self.vocab['UNK']))
            batch_input.append(new_line)
        for line in output:  # 依据Tag词典，将output转化为向量
            new_line = []
            for tag in line.strip().split():
                new_line.append(
                    int(self.tag_index_dict[tag]) if tag in self.tag_index_dict else print(
                        "There is a wrong Tag in {}!".format(line)))
            batch_output.append(new_line)

        return batch_input, batch_output

    def padding(self, batch_input, batch_output):
        # 为了满足pack_padded_sequence的要求，将batch数据按input数据的长度排序
        batch_data = list(zip(batch_input, batch_output))
        batch_input_data, batch_output_data = zip(*batch_data)
        in_out_pairs = []  # [[batch_input_data_item, batch_output_data_item], ...]
        for n in range(len(batch_input_data)):
            new_pair = []
            new_pair.append(batch_input_data[n])
            new_pair.append(batch_output_data[n])
            in_out_pairs.append(new_pair)
        in_out_pairs = sorted(in_out_pairs, key=lambda x: len(x[0]), reverse=True)  # 根据每对数据中第1个数据的长度排序

        # 排序后的数据重新组织成batch_x, batch_y
        batch_x = []
        batch_y = []
        for each_pair in in_out_pairs:
            batch_x.append(each_pair[0])
            batch_y.append(each_pair[1])

        # 按最长的数据pad
        batch_x_padded, x_lengths_original = self.pad_seq(batch_x)
        batch_y_padded, y_lengths_original = self.pad_seq(batch_y)

        return batch_x_padded, x_lengths_original, batch_y_padded, y_lengths_original

    def pad_seq(self, seq):
        seq_lengths = [len(st) for st in seq]
        # create an empty matrix with padding tokens
        pad_token = self.padding_idx
        longest_sent = max(seq_lengths)
        batch_size = len(seq)
        padded_seq = np.ones((batch_size, longest_sent)) * pad_token
        # copy over the actual sequences
        for i, x_len in enumerate(seq_lengths):
            sequence = seq[i]
            padded_seq[i, 0:x_len] = sequence[:x_len]
        return padded_seq, seq_lengths

    def prtplot(self, y_axis_values):
        x_axis_values = np.arange(1, len(y_axis_values) + 1, 1)
        y_axis_values = np.array(y_axis_values)
        plt.plot(x_axis_values, y_axis_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Base_Model!')
