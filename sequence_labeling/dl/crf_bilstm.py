#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
BiLSTM_CRF
======
A class for BiLSTM_CRF.
配置文件：cmed.dl.bilstm_crf.norm.json
"""

import argparse
import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.data_loader import DataLoader
from sequence_labeling.data_processor import DataProcessor
from sequence_labeling.utils.evaluate import Evaluator
from sequence_labeling.utils.evaluate_2 import Metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class CRF_BiLSTM(nn.Module):

    def __init__(self, **config):
        super(CRF_BiLSTM, self).__init__()
        self.config = config
        self.model_name = config['model_name']
        self.data_root = config['data_root']
        self.model_suffix = config['model_save_name']
        self.model_path = './model/{}_{}'.format(self.model_name, self.model_suffix)

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

        # CRF
        # 转移矩阵，矩阵中代表由状态j到i的概率
        self.transitions = nn.Parameter(
            torch.randn(self.tags_size, self.tags_size)).to(device)
        # 规则：任何状态都不能转移到'SOS','EOS'不能转移到其他状态
        self.transitions.data[self.tag_index_dict["SOS"], :] = -10000
        self.transitions.data[:, self.tag_index_dict["EOS"]] = -10000

        self.word_embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim,
                                            padding_idx=self.padding_idx)

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.layers,
                            bidirectional=self.bidirectional, batch_first=True)

        # 将模型输出映射到标签空间
        self.output_to_tag = nn.Linear(self.hidden_dim * self.n_directions, self.tags_size)

    def _init_hidden(self, batch_size):  # 初始化h_0 c_0
        hidden = (torch.zeros(self.layers * self.n_directions, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.layers * self.n_directions, batch_size, self.hidden_dim).to(device))
        return hidden

    def forward(self, x, x_lengths, y):
        batch_size, seq_len = x.size()
        hidden = self._init_hidden(batch_size)
        embeded = self.word_embeddings(x).to(device)
        embeded = rnn_utils.pack_padded_sequence(embeded, x_lengths, batch_first=True)
        output, _ = self.lstm(embeded, hidden)  # 使用初始化值
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        out = output.reshape(-1, output.shape[2])  # [batch_size*seq_len, tags_size]
        out = self.output_to_tag(out)

        crf_score, crf_seqs_tag = self.viterbi_decode(out)

        return out, crf_score, crf_seqs_tag

    def run_train(self, model, data_path):
        print('Running {} model. Training...'.format(self.model_name))
        run_mode = 'train'
        model.to(device)
        model.train()
        optimizer = self.optimizer_dict[self.optimizer_name](model.parameters(), lr=self.learning_rate)

        train_losses = []  # 记录每一batch的loss
        valid_losses = []
        valid_f1_scores = []
        best_valid_loss = float('inf')
        torch.backends.cudnn.enabled = False
        for epoch in range(self.num_epochs):
            train_loss = 0
            train_data_num = 0  # 统计数据量
            for seq_list, tag_list in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                               run_mode=run_mode):
                batch_input, batch_output = self.data_to_index(seq_list, tag_list)
                x, x_len, y, y_len = self.padding(batch_input, batch_output)
                train_data_num += len(x)
                batch_x = torch.tensor(x).long().to(device)
                batch_y = torch.tensor(y).long().to(device)

                tag_scores, _, _ = model(batch_x, x_len, batch_y)
                batch_y = batch_y.view(-1)
                loss = self.neg_log_likelihood(tag_scores, batch_y).sum()

                model.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_losses.append(train_loss / train_data_num)  # 记录评价结果

            # 模型验证
            avg_eval_loss, eval_f1 = self.evaluate(model, data_path)
            print("Done Epoch{}. Eval_Loss={}   Eval_f1={}".format(epoch + 1, avg_eval_loss, eval_f1))
            valid_losses.append(avg_eval_loss)
            valid_f1_scores.append(eval_f1)

            # 保存参数最优的模型
            if avg_eval_loss < best_valid_loss:
                best_valid_loss = avg_eval_loss
                torch.save(model, '{}'.format(self.model_path))

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
        model = torch.load(self.model_path)

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
        model.to(device)
        model.eval()
        with torch.no_grad():
            total_loss = 0
            data_num = 0
            all_y_predict = []
            all_y_true = []
            for seq_list, tag_list in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                               run_mode=run_mode):
                batch_input, batch_output = self.data_to_index(seq_list, tag_list)
                x, x_len, y, y_len = self.padding(batch_input, batch_output)
                data_num += len(x)
                batch_x = torch.tensor(x).long().to(device)
                batch_y = torch.tensor(y).long().to(device)
                batch_y = batch_y.view(-1)

                tag_scores, crf_score, crf_seqs_tag = model(batch_x, x_len, batch_y)

                loss = self.neg_log_likelihood(tag_scores, batch_y).sum()
                total_loss += loss.item()

                y_predict = list(crf_seqs_tag)
                y_predict = self.index_to_tag(y_predict, y_len)
                all_y_predict = all_y_predict + y_predict
                # print(list(set(all_y_predict)))

                y_true = y.flatten()
                y_true = self.index_to_tag(y_true, y_len)
                all_y_true = all_y_true + y_true
                # print(list(set(all_y_true)))

            return total_loss / data_num, all_y_true, all_y_predict

    def index_to_tag(self, y, y_len):
        index_tag_dict = dict(zip(self.tag_index_dict.values(), self.tag_index_dict.keys()))
        y_tagseq = []
        seq_len = int(len(y) / len(y_len))
        for i in range(len(y_len)):
            temp_list = y[i * seq_len:(i + 1) * seq_len]
            for count in range(seq_len):
                if count < y_len[i]:
                    y_tagseq.append(index_tag_dict[temp_list[count]])
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
        batch_x_padded, x_lengths_original = self.pad_seq(batch_x, self.padding_idx)
        batch_y_padded, y_lengths_original = self.pad_seq(batch_y, self.padding_idx_tags)

        return batch_x_padded, x_lengths_original, batch_y_padded, y_lengths_original

    def pad_seq(self, seq, padding_idx):
        seq_lengths = [len(st) for st in seq]
        # create an empty matrix with padding tokens
        pad_token = padding_idx
        longest_sent = max(seq_lengths)
        batch_size = len(seq)
        padded_seq = np.ones((batch_size, longest_sent)) * pad_token
        # copy over the actual sequences
        for i, x_len in enumerate(seq_lengths):
            sequence = seq[i]
            padded_seq[i, 0:x_len] = sequence[:x_len]
        return padded_seq, seq_lengths

    # mask padding
    def generate_masks(self, sequence_tensor, sequence_lengths):
        batch_size, seq_len = sequence_tensor.shape
        seq_masks = torch.BoolTensor(batch_size, seq_len)
        for seq_id, src_len in enumerate(sequence_lengths):
            seq_masks[seq_id, :src_len] = True
        return seq_masks

    def prtplot(self, y_axis_values):
        x_axis_values = np.arange(1, len(y_axis_values) + 1, 1)
        y_axis_values = np.array(y_axis_values)
        plt.plot(x_axis_values, y_axis_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.show()

    # CRF
    # 计算配分函数Z(x)
    # Z(x)的作用：做全局归一化，解决标注偏置问题。其关键在于需要遍历所有路径。
    # Z(x)的计算：前向算法
    def forward_alg(self, feats):
        init_alphas = torch.full((1, self.tags_size), -10000.).to(device)
        # SOS_TAG has all of the score.
        init_alphas[0][self.tag_index_dict["SOS"]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # 初始状态的forward_var
        forward_var = init_alphas

        # 迭代整个句子
        for feat in feats:  # feats为输入序列，即x
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tags_size):  # 6
                # broadcast the emission score: it is the same regardless of
                # the previous tag 1*6 emit_score的6个值相同
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tags_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i 1*6 从i到下一个tag的概率
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_index_dict["EOS"]]  # 到第（t-1）step时6个标签的各自分数
        alpha = log_sum_exp(terminal_var)
        return alpha

    # 计算给定输入序列和标签序列的匹配函数，即s(x,y)
    def score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(device)
        tags = torch.cat(
            [torch.tensor([self.tag_index_dict["SOS"]], dtype=torch.long).to(device), tags])  # 将SOS_TAG的标签拼接到tag序列上
        for i, feat in enumerate(feats):
            # self.transitions[tags[i + 1], tags[i]]：第i时间步对应的标签转移到下一时间步对应标签的概率
            # feat[tags[i+1]]：feats第i个时间步对应标签的score。之所以用i+1是要跳过tag序列开头的SOS
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_index_dict["EOS"], tags[-1]]
        return score

    # 维特比解码，给定输入x和相关参数(发射矩阵和转移矩阵)，获得概率最大的标签序列
    def viterbi_decode(self, feats):  # 维特比
        backpointers = []

        # Initialize the viterbi variables in log space 初始化
        init_vvars = torch.full((1, self.tags_size), -10000.).to(device)
        init_vvars[0][self.tag_index_dict["SOS"]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tags_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]  # 上一时刻的forward_var与transition矩阵相作用
                best_tag_id = argmax(next_tag_var)  # 选取上一时刻的tag使得到当前时刻的某个tag的路径分数最大
                bptrs_t.append(best_tag_id)  # 添加路径,注意此时的best_tag_id指向的是上一时刻的label
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)  # 从step0到step(i-1)时5个序列中每个序列的最大score
            backpointers.append(bptrs_t)

        # Transition to EOS_TAG
        terminal_var = forward_var + self.transitions[self.tag_index_dict["EOS"]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):  # 从后向前走，找到一个best路径
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the SOS tag (we dont want to return that to the caller)
        SOS = best_path.pop()
        assert SOS == self.tag_index_dict["SOS"]  # Sanity check
        best_path.reverse()  # 把从后向前的路径正过来
        return path_score, best_path

    # 损失函数 = Z(x) - s(x,y)
    def neg_log_likelihood(self, feats, tags):
        forward_score = self.forward_alg(feats)  # 计算配分函数Z(x)
        gold_score = self.score_sentence(feats, tags)  # 根据真实的tags计算score
        return forward_score - gold_score


def argmax(vec):
    # return the argmax as a python int
    # 返回行tensor最大值
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
# 计算离差平方和
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


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
