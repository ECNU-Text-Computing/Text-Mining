#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
svm
======
A class for something.
"""

import os
import sys
import argparse
import datetime
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from text_classification.shallow.base_model import BaseModel


class LR(BaseModel):
    def __init__(self, metrics_num):
        super(LR, self).__init__()
        self.model_name = 'lr'

    def build(self):
        self.model = LogisticRegression()


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

    print('Done svm!')
