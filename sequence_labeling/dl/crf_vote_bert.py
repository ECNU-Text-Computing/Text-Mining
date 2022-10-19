#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: CRF_Vote_BERT
======
配置文件：cmed.dl.crf_vote_bert.norm.json

配置文件中"checkpoint"建议使用:
hfl/chinese-bert-wwm-ext
hfl/chinese-roberta-wwm-ext
hfl/chinese-macbert-base
"""

import argparse
import datetime
import sys

import torch

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.data_loader import DataLoader
from sequence_labeling.utils.evaluate import Evaluator
from sequence_labeling.utils.evaluate_2 import Metrics
from sequence_labeling.dl.crf_nn_bert import CRF_NN_BERT
from sequence_labeling.dl.crf_lstm_bert import CRF_LSTM_BERT
from sequence_labeling.dl.crf_cnn_bert import CRF_CNN_BERT
from sequence_labeling.dl.crf_mhsa_bert import CRF_MHSA_BERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


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


class CRF_Vote_BERT(CRF_NN_BERT):
    def __init__(self, **config):
        super().__init__(**config)
        self.model_mlp = CRF_NN_BERT(**config)
        self.model_bilstm = CRF_LSTM_BERT(**config)
        self.model_cnn = CRF_CNN_BERT(**config)
        self.model_self_attn = CRF_MHSA_BERT(**config)

        self.model_mlp_path = './model/vote_mlp.ckpt'
        self.model_bilstm_path = './model/vote_bilstm.ckpt'
        self.model_cnn_path = './model/vote_cnn.ckpt'
        self.model_self_attn_path = './model/vote_self_attn.ckpt'

    def run_train(self, model, data_path):
        print('Running {} model. Training...'.format(self.model_name))
        run_mode = 'train'
        self.model_mlp.to(device)
        self.model_bilstm.to(device)
        self.model_cnn.to(device)
        self.model_self_attn.to(device)

        mlp_optimizer = self.optimizer_dict[self.optimizer_name](self.model_mlp.parameters(), lr=self.learning_rate)
        bilstm_optimizer = self.optimizer_dict[self.optimizer_name](self.model_bilstm.parameters(),
                                                                    lr=self.learning_rate)
        cnn_optimizer = self.optimizer_dict[self.optimizer_name](self.model_cnn.parameters(), lr=self.learning_rate)
        self_attn_optimizer = self.optimizer_dict[self.optimizer_name](self.model_self_attn.parameters(),
                                                                       lr=self.learning_rate)

        self.model_mlp.train()
        self.model_bilstm.train()
        self.model_cnn.train()
        self.model_self_attn.train()

        train_losses = []  # 记录每一batch的loss
        valid_losses = []
        valid_f1_scores = []
        best_valid_loss = float('inf')
        best_valid_f1 = float('-inf')
        best_mlp_f1 = float('-inf')
        best_bilstm_f1 = float('-inf')
        best_cnn_f1 = float('-inf')
        best_sa_f1 = float('-inf')
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

                # 运行models
                mlp_tag_scores, _, _ = self.model_mlp(seq_list)
                mlp_loss = self.neg_log_likelihood(mlp_tag_scores, batch_y).sum()
                self.model_mlp.zero_grad()
                mlp_loss.backward()
                mlp_optimizer.step()

                bilstm_tag_scores, _, _ = self.model_bilstm(seq_list)
                bilstm_loss = self.neg_log_likelihood(bilstm_tag_scores, batch_y).sum()
                self.model_bilstm.zero_grad()
                bilstm_loss.backward()
                bilstm_optimizer.step()

                cnn_tag_scores, _, _ = self.model_cnn(seq_list)
                cnn_loss = self.neg_log_likelihood(cnn_tag_scores, batch_y).sum()
                self.model_cnn.zero_grad()
                cnn_loss.backward()
                cnn_optimizer.step()

                self_attn_tag_scores, _, _ = self.model_self_attn(seq_list)
                self_attn_loss = self.neg_log_likelihood(self_attn_tag_scores, batch_y).sum()
                self.model_self_attn.zero_grad()
                self_attn_loss.backward()
                self_attn_optimizer.step()

                train_loss += (mlp_loss.item() + bilstm_loss.item() + cnn_loss.item() + self_attn_loss.item())
                # print(train_data_num)

            train_losses.append(train_loss / train_data_num)  # 记录评价结果

            # 模型验证
            avg_eval_loss, eval_f1, mlp_f1, bilstm_f1, cnn_f1, sa_f1 = self.vote_evaluate(self.model_mlp,
                                                                                          self.model_bilstm,
                                                                                          self.model_cnn,
                                                                                          self.model_self_attn,
                                                                                          data_path)
            print("Done Epoch{}. Eval_Loss={}  Eval_f1={}  mlp_lstm_cnn_attn={} / {} / {} / {}".format(epoch + 1,
                                                                                                       round(avg_eval_loss, 4),
                                                                                                       round(eval_f1, 4),
                                                                                                       round(mlp_f1, 4),
                                                                                                       round(bilstm_f1, 4),
                                                                                                       round(cnn_f1, 4),
                                                                                                       round(sa_f1, 4)))
            valid_losses.append(avg_eval_loss)
            valid_f1_scores.append(eval_f1)

            # 保存参数最优的模型
            if eval_f1 > best_valid_f1:
                best_valid_f1 = eval_f1
            if mlp_f1 > best_mlp_f1:
                torch.save(self.model_mlp, self.model_mlp_path)
            if bilstm_f1 > best_bilstm_f1:
                torch.save(self.model_bilstm, self.model_bilstm_path)
            if cnn_f1 > best_cnn_f1:
                torch.save(self.model_cnn, self.model_cnn_path)
            if sa_f1 > best_sa_f1:
                torch.save(self.model_self_attn, self.model_self_attn_path)

        return train_losses, valid_losses, valid_f1_scores

    def vote_evaluate(self, model_mlp, model_bilstm, model_cnn, model_sa, data_path):
        run_mode = 'eval'

        avg_loss, y_true, y_predict, mlp_predict, bilstm_predict, cnn_predict, self_attn_predict = self.vote_eval_process(
            model_mlp, model_bilstm, model_cnn, model_sa,
            run_mode, data_path)
        f1_score = Evaluator().f1score(y_true, y_predict)

        # sub_models f1 scores
        mlp_f1 = Evaluator().f1score(y_true, mlp_predict)
        bilstm_f1 = Evaluator().f1score(y_true, bilstm_predict)
        cnn_f1 = Evaluator().f1score(y_true, cnn_predict)
        sa_f1 = Evaluator().f1score(y_true, self_attn_predict)

        return avg_loss, f1_score, mlp_f1, bilstm_f1, cnn_f1, sa_f1

    def test(self, data_path):
        print('Running {} model. Testing data in folder: {}'.format(self.model_name, data_path))
        run_mode = 'test'

        model_mlp = torch.load(self.model_mlp_path)
        model_bilstm = torch.load(self.model_bilstm_path)
        model_cnn = torch.load(self.model_cnn_path)
        model_self_attn = torch.load(self.model_self_attn_path)

        avg_loss, y_true, y_predict, mlp_predict, bilstm_predict, cnn_predict, self_attn_predict = self.vote_eval_process(
            model_mlp, model_bilstm, model_cnn, model_self_attn, run_mode, data_path)

        # 输出评价结果
        print(Evaluator().classifyreport(y_true, y_predict))
        f1_score = Evaluator().f1score(y_true, y_predict)

        # 输出混淆矩阵
        metrix = Metrics(y_true, y_predict)
        metrix.report_scores()
        metrix.report_confusion_matrix()

        # sub_models f1 scores
        mlp_f1 = Evaluator().f1score(y_true, mlp_predict)
        bilstm_f1 = Evaluator().f1score(y_true, bilstm_predict)
        cnn_f1 = Evaluator().f1score(y_true, cnn_predict)
        sa_f1 = Evaluator().f1score(y_true, self_attn_predict)

        return f1_score, mlp_f1, bilstm_f1, cnn_f1, sa_f1

    def vote_eval_process(self, model_mlp, model_bilstm, model_cnn, model_sa, run_mode, data_path):
        model_mlp = model_mlp.to(device)
        model_bilstm = model_bilstm.to(device)
        model_cnn = model_cnn.to(device)
        model_self_attn = model_sa.to(device)

        model_mlp.eval()
        model_bilstm.eval()
        model_cnn.eval()
        model_self_attn.eval()

        with torch.no_grad():
            total_loss = 0
            data_num = 0
            all_y_predict = []
            all_mlp_predict = []
            all_bilstm_predict = []
            all_cnn_predict = []
            all_self_attn_predict = []
            all_y_true = []

            for seq_list, tag_list in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                               run_mode=run_mode):
                data_num += len(seq_list)
                # 生成batch_y
                batch_output = self.tag_to_index(tag_list)
                y = self.pad_taglist(batch_output)
                batch_y = torch.tensor(y).long().to(device)
                batch_y = batch_y.view(-1)

                # 运行models
                mlp_tag_scores, _, mlp_seqs_tag = model_mlp(seq_list)
                mlp_loss = self.neg_log_likelihood(mlp_tag_scores, batch_y).sum()

                bilstm_tag_scores, _, bilstm_seqs_tag = model_bilstm(seq_list)
                bilstm_loss = self.neg_log_likelihood(bilstm_tag_scores, batch_y).sum()

                cnn_tag_scores, _, cnn_seqs_tag = model_cnn(seq_list)
                cnn_loss = self.neg_log_likelihood(cnn_tag_scores, batch_y).sum()

                self_attn_tag_scores, _, sa_seqs_tag = model_self_attn(seq_list)
                self_attn_loss = self.neg_log_likelihood(self_attn_tag_scores, batch_y).sum()

                total_loss += (mlp_loss.item() + bilstm_loss.item() + cnn_loss.item() + self_attn_loss.item())

                # y_predict of each model
                mlp_y_predict = list(mlp_seqs_tag)
                mlp_y_predict = self.index_to_tag(mlp_y_predict, self.list_to_length(tag_list))
                all_mlp_predict = all_mlp_predict + mlp_y_predict

                bilstm_y_predict = list(bilstm_seqs_tag)
                bilstm_y_predict = self.index_to_tag(bilstm_y_predict, self.list_to_length(tag_list))
                all_bilstm_predict = all_bilstm_predict + bilstm_y_predict

                cnn_y_predict = list(cnn_seqs_tag)
                cnn_y_predict = self.index_to_tag(cnn_y_predict, self.list_to_length(tag_list))
                all_cnn_predict = all_cnn_predict + cnn_y_predict

                self_attn_y_predict = list(sa_seqs_tag)
                self_attn_y_predict = self.index_to_tag(self_attn_y_predict, self.list_to_length(tag_list))
                all_self_attn_predict = all_self_attn_predict + self_attn_y_predict

                y_true = y.flatten()
                y_true = self.index_to_tag(y_true, self.list_to_length(tag_list))
                all_y_true = all_y_true + y_true
                # print(list(set(all_y_true)))

            # 投票表决
            mlp_len = len(all_mlp_predict)
            bilstm_len = len(all_bilstm_predict)
            cnn_len = len(all_cnn_predict)
            self_attn_len = len(all_self_attn_predict)
            if mlp_len == bilstm_len and bilstm_len == cnn_len and cnn_len == self_attn_len:
                for i in range(mlp_len):
                    labels = []
                    labels.append(all_mlp_predict[i])
                    labels.append(all_bilstm_predict[i])
                    labels.append(all_cnn_predict[i])
                    labels.append(all_self_attn_predict[i])

                    unique_labels = set(labels)  # set集合去重
                    label_num = {}
                    for label in unique_labels:
                        label_num[label] = labels.count(label)

                    # 对字典按value排序，返回值为list
                    label_num_sorted = sorted(label_num.items(), key=lambda x: x[1], reverse=True)

                    if len(label_num_sorted) == 1 or len(label_num_sorted) == 3 or (
                            len(label_num_sorted) == 2 and label_num_sorted[0][1] > label_num_sorted[1][1]):
                        all_y_predict.append(label_num_sorted[0][0])
                    else:
                        all_y_predict.append(all_bilstm_predict[i])
                # print(list(set(all_y_predict)))

            return total_loss / data_num, all_y_true, all_y_predict, all_mlp_predict, all_bilstm_predict, all_cnn_predict, all_self_attn_predict


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
