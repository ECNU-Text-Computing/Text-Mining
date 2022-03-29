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
from sklearn.svm import LinearSVC
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from text_classification.shallow.base_model import BaseModel


class SVM(BaseModel):
    def __init__(self, metrics_num):
        super(SVM, self).__init__()
        self.model_name = 'svm'

    def build(self):
        self.model = LinearSVC()


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
