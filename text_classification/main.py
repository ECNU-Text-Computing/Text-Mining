#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
main
======
A class for something.
"""

import os
import argparse
import datetime
import json

from data_loader import DataLoader
from shallow.logistic_regression import LR
from shallow.svm import SVM
from deep.base_model import BaseModel
from deep.text_cnn import TextCNN
from deep.bert import BERT
from deep.hierarchy.hierarchical_att import HierAttNet

# 保存机器学习模型的全局变量。
ml_model_dict = {
    'svm': SVM,
    'lr': LR
}

# 保存深度学习模型的全局变量。
dl_model_dict = {
    'mlp': BaseModel,
    'textcnn': TextCNN,
    'bert': BERT,
    'hierarchical_attention': HierAttNet
}


# 机器学习模型的mian函数。
def main_ml(config):
    # 解读config文件中的元素。
    # json数据加载（load）到程序中后，即为一个字典（dict）。
    data_name = config['data_name']  # 'aapr'
    model_name = config['model_name']  # 'svm'
    feature = config['feature']  # 'tf'
    metrics_num = config['metrics_num']

    # 数据导入类的实例化。
    data_loader = DataLoader()

    # 导入训练数据。
    x_train, y_train = data_loader.data_load(data_name=data_name, phase='train', feature=feature)
    # 模型类的实例化，具体选择哪个模型，由model_name决定。
    model = ml_model_dict[model_name](metrics_num=metrics_num)
    # 该方法又实例化了sklearn中的机器学习模型。
    model.build()

    # 用训练数据训练模型。
    model.train(x_train, y_train)

    # 将实验中间产生的数据，如训练好的模型文件，保存在实验（exp）文件夹中。
    # save_folder = 'exp/aapr/'
    save_folder = '{}{}/'.format(data_loader.exp_root, data_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # 最后，保存模型的文件为以下格式，例如：exp/aapr/lr.tf。
    model_path = "{}{}/{}.{}".format(data_loader.exp_root, data_name, model_name, feature)
    # 将训练好的模型保存到这个地址。
    model.save_model(model_path)

    # 评价模型在训练集上的效果。评价指标包括accuracy、precision、recall、f1 score等。
    model.evaluate(x_train, y_train, phase='train')

    # 导入验证数据集。
    x_val, y_val = data_loader.data_load(data_name=data_name, phase='val', feature=feature)
    # 导入测试数据集。
    x_test, y_test = data_loader.data_load(data_name=data_name, phase='test', feature=feature)

    # 将验证集的评价结果输出。
    model.evaluate(x_val, y_val, phase='val')
    # 将测试集的评价结果输出。
    sorted_cal_res = model.evaluate(x_test, y_test, phase='test')


def main_dl(config):
    data_name = config['data_name']  # aapr
    model_name = config['model_name']  # mlp
    feature = config['feature']  # hierarchical

    # 数据导入类的实例化。
    data_loader = DataLoader()



    # 导入字典。
    # 该字典可将每个字符映射为一个id，进而可将一个字符序列转化为一个id的序列。
    # 根据config中的data_name，选择对应数据的字典。
    # word_dict_path = "exp/aapr/vocab.cover1.min0.json"
    word_dict_path = "exp/{}/vocab.cover1.min1.json".format(data_name)
    with open(word_dict_path, 'r') as fp:
        word_dict = json.load(fp)
        print("Load sent dict from {}.".format(word_dict_path))

    # 训练/验证/测试.输入 数据集的地址。
    input_path_train = 'datasets/{}/train.input_data'.format(data_name)
    input_path_val = 'datasets/{}/val.input_data'.format(data_name)
    input_path_test = 'datasets/{}/test.input_data'.format(data_name)

    # 训练/验证/测试.输出 数据集的地址。
    output_path_train = 'datasets/{}/train.output'.format(data_name)
    output_path_val = 'datasets/{}/val.output'.format(data_name)
    output_path_test = 'datasets/{}/test.output'.format(data_name)

    # 将实验中间产生的数据，如训练好的模型文件，保存在实验（exp）文件夹中。
    # save_folder = 'exp/aapr/'
    save_folder = '{}{}/'.format(data_loader.exp_root, data_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # 深度学习实验的中间结果会保存在这个文件夹中。
    # save_folder = 'exp/aapr/dl/'
    save_folder = 'exp/{}/dl/'.format(data_name)
    # 如果你还没有创建这个文件夹，那下面的代码将帮你创建一个文件夹。
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # 使用不同模型的时候，后面的mlp要改为对应模型的名字。
    # save_model_folder = 'exp/aapr/dl/mlp/'
    save_model_folder = 'exp/{}/dl/{}/'.format(data_name, model_name)
    # 如果你还没有创建这个文件夹，那下面的代码将帮你创建一个文件夹。
    if not os.path.exists(save_model_folder):
        os.mkdir(save_model_folder)

    vocab_size = len(word_dict)
    # 实例化深度学习模型。具体实例化哪个模型由model_name决定。
    model = dl_model_dict[model_name](vocab_size=vocab_size, **config)

    # data_generator = 选择
    if feature == 'bert':
        generator = data_loader.bert_data_generator
    elif feature == 'hierarchical':
        generator = data_loader.hierarchical_data_generator
    else:
        generator = data_loader.data_generator

    # 调用这个类的train_model函数来训练这个模型。
    model.train_model(model, feature, generator, input_path_train, output_path_train, word_dict,
                      input_path_val=input_path_val, output_path_val=output_path_val,
                      input_path_test=input_path_test, output_path_test=output_path_test,
                      save_folder=save_model_folder)


# 对任何Python脚本的入口，都在main这里。
if __name__ == '__main__':
    # 用来记录程序运行的时间。
    # 这里是开始的时间。
    start_time = datetime.datetime.now()

    # 从程序外部——终端（terminal）——传入一些参数。
    # 这里是为了灵活的选择想要训练的模型和所用的参数文件。
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')
    # args存储了所有从外部传入的参数，可以将之理解为一个dict。
    args = parser.parse_args()

    # 模型参数设置。
    # 深度学习模型的参数。
    # 此处args.phase = aapr.dl.mlp.norm
    # config_path = './config/aapr/dl/aapr.dl.mlp.norm.json'
    config_path = './config/{}/{}/{}.json'.format(args.phase.strip().split('.')[0],
                                                  args.phase.strip().split('.')[1],
                                                  args.phase.strip())
    # 机器学习模型的参数。
    # 此处args.phase = aapr.ml.svm.tf
    # config_path = './config/aapr/ml/aapr.ml.svm.tf.json'

    if not os.path.exists(config_path):
        raise RuntimeError("There is no {} config.".format(args.phase))

    # 用json读取config的参数。
    config = json.load(open(config_path, 'r'))
    print('config: ', config)

    # 判断是机器学习模型还是深度学习模型，其中ml_model_dict和dl_model_dict是程序最前面定义的全局变量。
    # ml_model_dict保存了所有已定义的机器学习模型；
    # dl_model_dict保存了所有已定义的深度学习模型。
    model_name = config['model_name']
    # 如果是
    if model_name in ml_model_dict:
        # 运行机器学习的main函数。
        main_ml(config)
    elif model_name in dl_model_dict:
        # 运行深度学习的main函数。
        main_dl(config)
    else:
        raise RuntimeError("There is no model name.".format(model_name))

    # 程序结束时的时间。
    end_time = datetime.datetime.now()
    # 计算程序总共运行的时间，单位是秒（s）。
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))
    print('Done main!')
