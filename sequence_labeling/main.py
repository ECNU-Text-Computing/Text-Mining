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

from dl.base_model import BaseModel
from dl.bilstm import BiLSTM
from dl.bilstm_crf import BiLSTM_CRF
from dl.gru import GRU
from dl.s2s import SeqToSeq
from dl.s2s_dotproduct_attn import SeqToSeq_DotProductAttn

ml_model_dict = {
}

dl_model_dict = {
    'LSTM': BaseModel,
    'BiLSTM': BiLSTM,
    'GRU': GRU,
    'BiLSTM_CRF': BiLSTM_CRF,
    'SeqToSeq': SeqToSeq,
    'SeqToSeq_DotProduct_Attn': SeqToSeq_DotProductAttn
}


def main_dl(config):
    if config['n_folds'] == 0:
        model = dl_model_dict[model_name](**config)
        print(model)
        # model.run_model(model, 'train')
        # model.run_model(model, 'test')
        model.run_train(model)
        model.test()
    elif config['n_folds'] > 0:
        for n in range(config['n_folds']):
            model = dl_model_dict[model_name](**config)
            model.run_train(model)
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
    # args = 'cmed.dl.base_model.norm'
    # config_path = './config/cmed/dl/cmed.dl.base_model.norm.json'
    if not os.path.exists(config_path):
        raise RuntimeError("There is no {} config.".format(args))
    config = json.load(open(config_path, 'r'))
    # print('config: ', config)

    model_name = config['model_name']
    if model_name in dl_model_dict:
        main_dl(config)
    else:
        raise RuntimeError("There is no model name.".format(model_name))

    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args, (end_time - start_time).seconds))
    print('Done main!')
