#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
main
======
The main class for sequence labeling.
"""

import argparse
import datetime
import json
import os
import sys
from utils.logger import Logger

from dl.base_model import BaseModel
from dl.bilstm import BiLSTM
from dl.bilstm_crf import BiLSTM_CRF
from dl.gru import GRU
from dl.rnn import RNN
from dl.s2s import SeqToSeq
from dl.s2s_dotproduct_attn import SeqToSeq_DotProductAttn
from dl.self_attention import Self_Attention
from dl.self_attention_multihead import Self_Attention_Multi_Head
from dl.transformer import Transformer
from dl.mlp import MLP
from dl.cnn import CNN
from dl.bert_mlp import Bert_MLP
from dl.bert_lstm import Bert_LSTM
from dl.bert_bilstm import Bert_BiLSTM
from dl.bert_cnn import Bert_CNN
from dl.bert_gru import Bert_GRU
from dl.bert_rnn import Bert_RNN
from dl.bert_s2s import Bert_S2S
from dl.bert_s2s_attn import Bert_S2S_Attn
from dl.bert_self_attn import Bert_Self_Attn
from dl.bert_self_attn_multihead import Bert_Self_Attn_Multihead

ml_model_dict = {
}

dl_model_dict = {
    'LSTM': BaseModel,
    'BiLSTM': BiLSTM,
    'RNN': RNN,
    'GRU': GRU,
    'BiLSTM_CRF': BiLSTM_CRF,
    'MLP': MLP,
    'CNN': CNN,
    'Self_Attn': Self_Attention,
    'Self_Attn_Multi_Head': Self_Attention_Multi_Head,
    'SeqToSeq': SeqToSeq,
    'SeqToSeq_DotProductAttn': SeqToSeq_DotProductAttn,
    'Transformer': Transformer,
    'Bert_MLP': Bert_MLP,
    'Bert_LSTM': Bert_LSTM,
    'Bert_BiLSTM': Bert_BiLSTM,
    'Bert_CNN': Bert_CNN,
    'Bert_GRU': Bert_GRU,
    'Bert_RNN': Bert_RNN,
    'Bert_S2S': Bert_S2S,
    'Bert_S2S_Attn': Bert_S2S_Attn,
    'Bert_Self_Attn': Bert_Self_Attn,
    'Bert_Self_Attn_Multihead': Bert_Self_Attn_Multihead
}


def main_dl(config):
    if config['n_folds'] == 0:
        model = dl_model_dict[model_name](**config)
        print(model)
        data_path = '{}0/'.format(config['data_root'])
        train_losses, eval_losses, eval_f1_scores = model.run_train(model, data_path)
        test_f1_score = model.test(data_path)
        print("train_losses: {}\n, eval_losses: {}\n, eval_f1_scores: {}\n, test_f1_score: {}\n".format(train_losses,
                                                                                                        eval_losses,
                                                                                                        eval_f1_scores,
                                                                                                        test_f1_score))
    elif config['n_folds'] > 0:
        train_results, eval_results, eval_f1, test_f1 = [], [], [], []
        for n in range(config['n_folds']):
            print("{} : This is the {} cross fold.".format(config['model_name'], n))
            model = dl_model_dict[model_name](**config)
            if n == 0:
                print(model)
            data_path = '{}{}/'.format(config['data_root'], n)
            train_losses, eval_losses, eval_f1_scores = model.run_train(model, data_path)
            test_f1_score = model.test(data_path)

            train_results.append(train_losses)
            eval_results.append(eval_losses)
            eval_f1.append(eval_f1_scores)
            test_f1.append(test_f1_score)
        print("train_losses: {}\n, eval_losses: {}\n, eval_f1_scores: {}\n, test_f1_score: {}\n".format(train_results,
                                                                                                        eval_results,
                                                                                                        eval_f1,
                                                                                                        test_f1))
    else:
        raise RuntimeError("There is no n_folds in config.")


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')
    args = parser.parse_args()
    print(args.phase)

    data_name = args.phase.strip().split('.')[0]
    model_cate = args.phase.strip().split('.')[1]
    config_path = './config/{}/{}/{}.json'.format(data_name, model_cate, args.phase)
    if not os.path.exists(config_path):
        raise RuntimeError("There is no {} config.".format(args))
    config = json.load(open(config_path, 'r'))

    model_name = config['model_name']
    # 重定向输出至文件和控制台
    sys.stdout = Logger('./log/{}.log'.format(model_name), sys.stdout)
    print('config: ', config)

    if model_name in dl_model_dict:
        main_dl(config)
    else:
        raise RuntimeError("There is no model name.".format(model_name))

    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args, (end_time - start_time).seconds))
    print('Done main!')
