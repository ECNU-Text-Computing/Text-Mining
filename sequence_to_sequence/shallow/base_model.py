#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
base_model
======
A class for something.
"""

import os
import sys
import argparse
import datetime
from joblib import dump, load
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from text_classification.utils.metrics import cal_all


class BaseModel(object):
    def __init__(self, metrics_num=4):
        print('Init...')
        self.model_name = 'ml'
        self.metrics_num = metrics_num

    def build(self):
        self.model = None

    def train(self, x, y):
        self.model.ffitit(x, y)
        return self.model

    def save_model(self, path):
        dump(self.model, path)
        print("Successfully save {} model to {}.".format(self.model_name, path))

    def load_model(self, path):
        model = load(path)
        return model

    def test(self, x, path=None):
        if path:
            self.model = self.load_model(path)
        y = self.model.predict(x)
        return y

    def evaluate(self, x, y, path=None, phase='train'):
        if path:
            self.model = self.load_model(path)
        pred_y = self.model.predict(x)
        metric = cal_all
        if self.metrics_num == 4:
            metric = cal_all
        cal_res = metric(y, pred_y)
        sorted_cal_res = sorted(cal_res.items(), key=lambda x: x[0])
        metric_names = '\t'.join([name_score[0][1:] for name_score in sorted_cal_res])
        metric_score = '\t'.join(['{:.2f}'.format(name_score[1]) for name_score in sorted_cal_res])
        if phase == 'train':
            print("{}\t{}\t{}".format(self.model_name, phase, metric_names))
        print("{}\t{}\t{}".format(self.model_name, phase, metric_score))
        return sorted_cal_res


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

    print('Done base_model!')
