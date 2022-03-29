#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
main
======
A class for something.
"""

import argparse
import datetime
import json
import os

from sequence_labeling.dl.base_model import BaseModel
# from Deep.BiLSTM_CNN import BiLSTM_CRF
from sequence_labeling.data_processor import DataProcessor

ml_model_dict = {
}

dl_model_dict = {
    'LSTM': BaseModel,
    # 'BiLSTM_CNN': BiLSTM_CRF
}


def main_dl(config):
    vocab_size = len(DataProcessor().load_vocab())

    model = dl_model_dict[model_name](vocab_size=vocab_size, **config)
    print(model)
    model.run_Model(model, op_mode='train')


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()
    print(args.phase)
    # args = 'cmed.dl.BaseModel.norm'
    data_name = args.phase.strip().split('.')[0]
    model_cate = args.phase.strip().split('.')[1]
    config_path = './config/{}/{}/{}.json'.format(data_name, model_cate, args.phase)
    if not os.path.exists(config_path):
        raise RuntimeError("There is no {} config.".format(args))
    config = json.load(open(config_path, 'r'))
    print('config: ', config)

    model_name = config['model_name']
    if model_name in dl_model_dict:
        main_dl(config)
    else:
        raise RuntimeError("There is no model name.".format(model_name))

    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args, (end_time - start_time).seconds))
    print('Done main!')
