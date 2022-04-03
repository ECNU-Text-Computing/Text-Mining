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

from dl.base_model import BaseModel
from data_processor import DataProcessor

ml_model_dict = {
}

dl_model_dict = {
    'LSTM': BaseModel,
    # 'BiLSTM_CNN': BiLSTM_CRF
}


def main_dl(config):
    if config['n_folds'] == 0:
        data_path = '{}'.format(config['data_root'])
        model = dl_model_dict[model_name](**config)
        print(model)
        model.run_model(model, run_mode='train', data_path=data_path)
        model.run_model(model, run_mode='test', data_path=data_path)
    elif config['n_folds'] > 0:
        for n in range(config['n_folds']):
            data_path = '{}{}/'.format(config['data_root'], n)
            model = dl_model_dict[model_name](**config)
            model.run_model(model, run_mode='train', data_path=data_path)
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
    print('config: ', config)

    model_name = config['model_name']
    if model_name in dl_model_dict:
        main_dl(config)
    else:
        raise RuntimeError("There is no model name.".format(model_name))

    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args, (end_time - start_time).seconds))
    print('Done main!')
