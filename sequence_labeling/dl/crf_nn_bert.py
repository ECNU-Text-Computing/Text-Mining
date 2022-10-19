#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: CRF_NN_BERT
======
Equal to CRF_MLP_BERT
配置文件：cmed.dl.crf_nn_bert.norm.json

配置文件中"checkpoint"可替换为如下列表中的值:
hfl/chinese-macbert-base
hfl/chinese-bert-wwm-ext
hfl/chinese-roberta-wwm-ext
"""

import argparse
import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, BertModel

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.data_loader import DataLoader
from sequence_labeling.data_processor import DataProcessor
from sequence_labeling.utils.evaluate import Evaluator
from sequence_labeling.utils.evaluate_2 import Metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class CRF_NN_BERT(nn.Module):
    def __init__(self, **config):
        super(CRF_NN_BERT, self).__init__()
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

        self.optimizer_dict = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW
        }
        self.optimizer_name = config['optimizer_name']
        if self.optimizer_name not in self.optimizer_dict:
            raise ValueError("There is no optimizer_name: {}.".format(self.optimizer_name))

        # Bert模型
        self.checkpoint = config['checkpoint']
        self.tokenizer = BertTokenizer.from_pretrained(self.checkpoint)
        model_config = BertConfig.from_pretrained(self.checkpoint)
        model_config.output_hidden_states = True
        model_config.output_attentions = True
        self.bert_model = BertModel.from_pretrained(self.checkpoint, config=model_config)
        self.concat_to_hidden = nn.Linear(model_config.hidden_size * 4, model_config.hidden_size)

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.fc = nn.Sequential(nn.Linear(self.embedding_dim, self.hidden_dim), nn.ReLU(), self.dropout)
        self.linear_output_to_tag = nn.Linear(self.hidden_dim, self.tags_size)

        # print('完成类{}的初始化'.format(self.__class__.__name__))

    def forward(self, seq_list):
        # BertModel embedding
        batch = self.tokenizer(seq_list, padding=True, truncation=True, return_tensors="pt").to(device)
        # get_token_embedding()的第2个参数用于选择embedding的方式
        # 0: 使用“last_hidden_state”
        # 1: 使用“initial embedding outputs”，即hidden_states[0]
        # 2: 使用隐藏层后4层输出的和
        # 3: 使用隐藏层后4层输出的concat
        embedded = self.get_token_embedding(batch, 0)

        # 从embedded中删除表示[CLS],[SEP]的向量
        embedded = self.del_special_token(seq_list, embedded)

        output = self.fc(embedded)
        output = self.linear_output_to_tag(output)
        output = output.reshape(-1, output.shape[2])

        tag_scores = F.log_softmax(output, dim=1)  # [batch_size*seq_len, tags_size]

        crf_score, seqs_tag = self.viterbi_decode(tag_scores)

        return tag_scores, crf_score, seqs_tag

    def run_train(self, model, data_path):
        print('Running {} model. Training...'.format(self.model_name))
        run_mode = 'train'
        model.to(device)
        optimizer = self.optimizer_dict[self.optimizer_name](model.parameters(), lr=self.learning_rate)
        model.train()

        train_losses = []  # 记录每一batch的loss
        valid_losses = []
        valid_f1_scores = []
        best_valid_loss = float('inf')
        best_valid_f1 = float('-inf')
        torch.backends.cudnn.enabled = False
        for epoch in range(self.num_epochs):
            train_loss = 0
            train_data_num = 0  # 统计数据量
            for seq_list, tag_list in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                               run_mode=run_mode):
                train_data_num += len(seq_list)
                # 生成batch_y
                batch_output = self.tag_to_index(tag_list)
                y = self.pad_taglist(batch_output)
                batch_y = torch.tensor(y).long().to(device)
                batch_y = batch_y.view(-1)

                # 运行model
                tag_scores, _, _ = model(seq_list)
                loss = self.neg_log_likelihood(tag_scores, batch_y).sum()

                model.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                # print(train_data_num)

            train_losses.append(train_loss / train_data_num)  # 记录评价结果

            # 模型验证
            avg_eval_loss, eval_f1 = self.evaluate(model, data_path)
            print("Done Epoch{}. Eval_Loss={}   Eval_f1={}".format(epoch + 1, avg_eval_loss, eval_f1))
            valid_losses.append(avg_eval_loss)
            valid_f1_scores.append(eval_f1)

            # 保存参数最优的模型
            if eval_f1 > best_valid_f1:
                best_valid_f1 = eval_f1
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

        return f1_score, y_true, y_predict

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
                data_num += len(seq_list)
                # 生成batch_y
                batch_output = self.tag_to_index(tag_list)
                y = self.pad_taglist(batch_output)
                batch_y = torch.tensor(y).long().to(device)
                batch_y = batch_y.view(-1)

                # 运行model
                tag_scores, crf_score, seqs_tag = model(seq_list)

                loss = self.neg_log_likelihood(tag_scores, batch_y).sum()
                total_loss += loss.item()

                y_predict = list(seqs_tag)
                y_predict = self.index_to_tag(y_predict, self.list_to_length(tag_list))
                all_y_predict = all_y_predict + y_predict
                # print(list(set(all_y_predict)))

                y_true = y.flatten()
                y_true = self.index_to_tag(y_true, self.list_to_length(tag_list))
                all_y_true = all_y_true + y_true
                # print(list(set(all_y_true)))

            return total_loss / data_num, all_y_true, all_y_predict

    # 根据列表内容，生成描述其成员长度的list
    def list_to_length(self, input):
        seq_list = []
        for line in input:
            seq_list.append(line.strip().split())
        lenth_list = [len(seq) for seq in seq_list]
        return lenth_list

    # 将index转换为token
    # def index_to_tag(self, y):
    #     index_tag_dict = dict(zip(self.tag_index_dict.values(), self.tag_index_dict.keys()))
    #     y_tagseq = []
    #     for i in range(len(y)):
    #         y_tagseq.append(index_tag_dict[y[i]])
    #     return y_tagseq

    # 将index转换为token，并依据y_len(原始数据长度)除去[PAD]
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

    def tag_to_index(self, output):
        batch_output = []
        for line in output:  # 依据Tag词典，将output转化为向量
            new_line = []
            for tag in line.strip().split():
                new_line.append(
                    int(self.tag_index_dict[tag]) if tag in self.tag_index_dict else print(
                        "There is a wrong Tag in {}!".format(line)))
            batch_output.append(new_line)

        return batch_output

    # 补齐每批tag列表数据
    def pad_taglist(self, taglist):
        seq_lengths = [len(st) for st in taglist]
        # create an empty matrix with padding tokens
        pad_token = self.padding_idx_tags
        longest_sent = max(seq_lengths)
        batch_size = len(taglist)

        # 不添加[CLS], [SEP]
        padded_seq = np.ones((batch_size, longest_sent)) * pad_token  # !!!

        # 添加[CLS], [SEP]
        # padded_seq = np.ones((batch_size, longest_sent+2)) * pad_token
        # copy over the actual sequences
        for i, x_len in enumerate(seq_lengths):
            sequence = taglist[i]

            # 不添加[CLS], [SEP]
            padded_seq[i, 0:x_len] = sequence[:x_len]  # !!!

            # 添加[CLS], [SEP]
            # padded_seq[i, 0] = self.tag_index_dict['[CLS]']
            # padded_seq[i, 1:x_len+1] = sequence[:x_len]
            # padded_seq[i, x_len+1] = self.tag_index_dict['[SEP]']
        return padded_seq

    def del_special_token(self, batch_list, batch_tensor):
        batch_size, seq_len, embed_dim = batch_tensor.shape
        seq_lengths = [int(len(st) / 2) for st in batch_list]

        # 按句取出，除去句中表示[CLS], [SEP]的向量
        tmp = []
        for i in range(batch_size):
            if seq_lengths[i] + 2 == seq_len:
                newt = batch_tensor[i][1:-1, :]
                tmp.append(newt)
            else:
                part1 = batch_tensor[i][1:seq_lengths[i] + 1, :]
                part2 = batch_tensor[i][seq_lengths[i] + 2:, :]
                newt = torch.cat((part1, part2), 0)
                tmp.append(newt)

        # 重新拼接为表示batch的张量
        new_batch_tensor = tmp[0].unsqueeze(0)
        if len(tmp) > 1:
            for i in range(len(tmp) - 1):
                new_batch_tensor = torch.cat((new_batch_tensor, tmp[i + 1].unsqueeze(0)), 0)

        return new_batch_tensor

    def get_token_embedding(self, batch, method):
        self.bert_model.to(device)
        # self.bert_model.eval()
        #
        # with torch.no_grad():
        if method == 0:
            # 使用“last_hidden_state”
            # last_hidden_state.shape = (batch_size, sequence_length, hidden_size)
            embedded = self.bert_model(**batch).last_hidden_state

        # 使用hidden_states。hidden_states是一个元组，第一个元素是embedding，其余元素是各层的输出，
        # 每个元素的形状是(batch_size, sequence_length, hidden_size)。
        elif method == 1:
            # 第1种方式：使用“initial embedding outputs”
            embedded = self.bert_model(**batch).hidden_states[0]

        elif method == 2:
            # 第2种方式：使用隐藏层输出sum last 4 layers
            hidden_states = self.bert_model(**batch).hidden_states

            layers = len(hidden_states)
            batch_size, seq_len, embed_dim = hidden_states[-1].size()
            # print(layers, batch_size, seq_len, embed_dim)

            batch_embedding = []
            for batch_i in range(batch_size):
                token_embeddings = []
                for token_i in range(seq_len):
                    hidden_layers = []
                    for layer_i in range(layers):
                        vec = hidden_states[layer_i][batch_i][token_i]
                        hidden_layers.append(vec)
                    token_embeddings.append(hidden_layers)
                # 连接最后四层 [number_of_tokens, 3072]
                # concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in
                #                               token_embeddings]
                # batch_embedding.append(torch.stack((concatenated_last_4_layers)))

                # 对最后四层求和 [number_of_tokens, 768]
                summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings]
                batch_embedding.append(torch.stack(summed_last_4_layers))

            embedded = torch.stack(batch_embedding)
        else:
            # 第3种方式：使用隐藏层输出concat last 4 layers
            hidden_states = self.bert_model(**batch).hidden_states

            layers = len(hidden_states)
            batch_size, seq_len, embed_dim = hidden_states[-1].size()
            # print(layers, batch_size, seq_len, embed_dim)

            batch_embedding = []
            for batch_i in range(batch_size):
                token_embeddings = []
                for token_i in range(seq_len):
                    hidden_layers = []
                    for layer_i in range(layers):
                        vec = hidden_states[layer_i][batch_i][token_i]
                        hidden_layers.append(vec)
                    token_embeddings.append(hidden_layers)
                # 连接最后四层 [number_of_tokens, 3072]
                concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer
                                              in
                                              token_embeddings]
                batch_embedding.append(torch.stack(concatenated_last_4_layers))

            embedded = self.concat_to_hidden(torch.stack(batch_embedding))

        return embedded

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
        # Pop off the SOS tag (we don't want to return that to the caller)
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
