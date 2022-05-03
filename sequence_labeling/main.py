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
from dl.s2s import SeqToSeq
from dl.s2s_dotproduct_attn import SeqToSeq_DotProductAttn
from dl.self_attention import Self_Attention
from dl.self_attention_multihead import Self_Attention_Multi_Head
from dl.transformer import Transformer
from dl.mlp import MLP
from dl.cnn import CNN

ml_model_dict = {
}

dl_model_dict = {
    'LSTM': BaseModel,
    'BiLSTM': BiLSTM,
    'GRU': GRU,
    'BiLSTM_CRF': BiLSTM_CRF,
    'MLP': MLP,
    'CNN': CNN,
    'Self_Attn': Self_Attention,
    'Self_Attn_Multi_Head': Self_Attention_Multi_Head,
    'SeqToSeq': SeqToSeq,
    'SeqToSeq_DotProductAttn': SeqToSeq_DotProductAttn,
    'Transformer': Transformer
}


def main_dl(config):
    if config['n_folds'] == 0:
        model = dl_model_dict[model_name](**config)
        print(model)
        model.run_train(model)
        model.test()
    elif config['n_folds'] > 0:
        for n in range(config['n_folds']):
            model = dl_model_dict[model_name](**config)
            print(model)
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
