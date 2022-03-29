#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
metrics
======
A class for something.
"""

import os
import sys
import argparse
import datetime

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


def cal_acc(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    return acc


def cal_precision(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    return precision


def cal_recall(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    return recall


def cal_f1score(y_true, y_pred):
    f1score = f1_score(y_true, y_pred)
    return f1score


def cal_logloss(y_true, y_proba):
    logloss = log_loss(y_true, y_proba)
    return logloss


def cal_all(y_true, y_pred):
    acc = cal_acc(y_true, y_pred)
    precision = cal_precision(y_true, y_pred)
    recall = cal_recall(y_true, y_pred)
    f1score = cal_f1score(y_true, y_pred)
    return {'1acc': acc*100, '2precision': precision*100, '3recall': recall*100, '4f1score': f1score*100}


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

    print('Done metrics!')
