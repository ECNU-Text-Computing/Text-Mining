#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
The main class for running CRF_Vote_BERT.
"""

import argparse
import datetime
import json
import os
import sys

from dl.crf_cnn_bert import CRF_CNN_BERT
from dl.crf_lstm_bert import CRF_LSTM_BERT
from dl.crf_mhsa_bert import CRF_MHSA_BERT
from dl.crf_nn_bert import CRF_NN_BERT
from dl.crf_vote_bert import CRF_Vote_BERT
from utils.logger import Logger
from utils.evaluate import Evaluator
from utils.evaluate_2 import Metrics

dl_model_dict = {
    'CRF_NN_BERT': CRF_NN_BERT,
    'CRF_LSTM_BERT': CRF_LSTM_BERT,
    'CRF_CNN_BERT': CRF_CNN_BERT,
    'CRF_MHSA_BERT': CRF_MHSA_BERT,
    'CRF_Vote_BERT': CRF_Vote_BERT
}


def vote(mlp_pred, bilstm_pred, cnn_pred, sa_pred, y_true):
    mlp_predict = mlp_pred
    bilstm_predict = bilstm_pred
    cnn_predict = cnn_pred
    self_attn_predict = sa_pred
    y_predict = []

    sub_models_f1 = []
    sub_models_f1.append(Evaluator().f1score(y_true, mlp_predict))
    sub_models_f1.append(Evaluator().f1score(y_true, bilstm_predict))
    sub_models_f1.append(Evaluator().f1score(y_true, cnn_predict))
    sub_models_f1.append(Evaluator().f1score(y_true, self_attn_predict))
    max_f1 = max(sub_models_f1)
    idx = sub_models_f1.index(max_f1)

    best_predict = mlp_predict
    if idx == 1:
        best_predict = bilstm_predict
    if idx == 2:
        best_predict = cnn_predict
    if idx == 3:
        best_predict = self_attn_predict

    # 投票表决
    for i in range(len(mlp_predict)):
        labels = []
        labels.append(mlp_predict[i])
        labels.append(bilstm_predict[i])
        labels.append(cnn_predict[i])
        labels.append(self_attn_predict[i])

        unique_labels = set(labels)  # set集合去重
        label_num = {}
        for label in unique_labels:
            label_num[label] = labels.count(label)

        # 对字典按value排序，返回值为list
        label_num_sorted = sorted(label_num.items(), key=lambda x: x[1], reverse=True)

        if len(label_num_sorted) == 1 or len(label_num_sorted) == 3 or (
                len(label_num_sorted) == 2 and label_num_sorted[0][1] > label_num_sorted[1][1]):
            y_predict.append(label_num_sorted[0][0])
        else:
            y_predict.append(best_predict[i])
        # print(list(set(y_predict)))

    return y_predict


def run_model(config, n):
    model = dl_model_dict[model_name](**config)
    print(config['model_name'])
    data_path = '{}{}/'.format(config['data_root'], n)
    train_losses, eval_losses, eval_f1_scores = model.run_train(model, data_path)
    test_f1_score, seqs_true, seqs_predict = model.test(data_path)
    print("train_losses: {}\n, eval_losses: {}\n, eval_f1_scores: {}\n, test_f1_score: {}\n".format(train_losses,
                                                                                                    eval_losses,
                                                                                                    eval_f1_scores,
                                                                                                    test_f1_score))

    return seqs_true, seqs_predict


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')
    args = parser.parse_args()

    # 重定向输出至文件和控制台
    sys.stdout = Logger(
        './log/CCKS2019_{}.log'.format(args.phase.strip().split('.')[2]), sys.stdout)

    n_fold = 10
    vote_f1_scores = []
    for n in range(0, n_fold):
        print("\n\n\n{}_fold begin_______________".format(n))
        sub_models = ['cmed.dl.crf_nn_bert.norm', 'cmed.dl.crf_lstm_bert.norm', 'cmed.dl.crf_cnn_bert.norm',
                      'cmed.dl.crf_mhsa_bert.norm']
        preds = []
        y_true = []
        for item in sub_models:
            args.phase = item
            print(args.phase)

            data_name = args.phase.strip().split('.')[0]
            model_cate = args.phase.strip().split('.')[1]
            config_path = './config/{}/{}/{}.json'.format(data_name, model_cate, args.phase)
            if not os.path.exists(config_path):
                raise RuntimeError("There is no {} config.".format(args))
            config = json.load(open(config_path, 'r'))
            print('config: ', config)

            model_name = config['model_name']
            if model_name in dl_model_dict:
                seqs_true, seqs_predict = run_model(config, n)
            else:
                raise RuntimeError("There is no model name.".format(model_name))
            end_time = datetime.datetime.now()
            print('{} takes {} seconds.'.format(args, (end_time - start_time).seconds))
            print('Done {}!\n\n'.format(model_name))

            if len(y_true) == 0:
                y_true = seqs_true
                preds.append(seqs_predict)
            else:
                if y_true == seqs_true:
                    if len(preds[0]) == len(seqs_predict):
                        preds.append(seqs_predict)
                    else:
                        raise RuntimeError("The length of predict is diffent in {}.".format(model_name))
                else:
                    raise RuntimeError("y_true is diffent in {}.".format(model_name))

        vote_predict = vote(preds[0], preds[1], preds[2], preds[3], y_true)

        # 输出评价结果
        print("\n\n********{}_fold Vote Result********".format(n))
        vote_f1 = Evaluator().f1score(y_true, vote_predict)
        print("Vote f1={}\n".format(vote_f1))
        print(Evaluator().classifyreport(y_true, vote_predict))

        # 输出混淆矩阵
        metrix = Metrics(y_true, vote_predict)
        metrix.report_scores()
        metrix.report_confusion_matrix()

        vote_f1_scores.append(vote_f1)

    print("\n\nVote Model {}_fold f1 scores:{}".format(n_fold, vote_f1_scores))
