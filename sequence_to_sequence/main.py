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
import torch.nn.functional as F

from data_loader import DataLoader
from base_model import BaseModel
from seq2seq import Seq2seq
from deep.text_encoder_rnn import EncoderRNN
from deep.text_decoder_rnn import DecoderRNN

# 保存深度学习模型的全局变量。
dl_model_dict = {
    'mlp': BaseModel,
    'seq2seq': Seq2seq
}

encoder_dict = {
    'rnn': EncoderRNN
}

decoder_dict = {
    'rnn': DecoderRNN
}


def main_dl(config):
    data_name = config['data_name']  # aapr
    model_name = config['model_name']  # mlp
    feature = config['feature']
    max_len = config['max_len']  # 256
    hidden_dim = config['hidden_dim']  # 64
    sos_id = config['sos_id']
    eos_id = config['eos_id']
    encoder_name = config['encoder_name']  # encoder_rnn
    decoder_name = config['decoder_name']  # decoder_rnn
    if 'input_dropout_rate' in config.keys():
            input_dropout_rate = config['input_dropout_rate']
    else:
        input_dropout_rate = 0
    if 'dropout_rate' in config.keys():
            dropout_rate = config['dropout_rate']
    else:
        dropout_rate = 0
    if 'num_layers' in config.keys():
            num_layers = config['num_layers']
    else:
        num_layers = 1
    if 'bidirectional' in config.keys():
            bidirectional = config['bidirectional']
    else:
        bidirectional = False
    if 'rnn_cell' in config.keys():
            rnn_cell = config['rnn_cell']
    else:
        rnn_cell = 'lstm'
    if 'variable_lengths' in config.keys():
            variable_lengths = config['variable_lengths']
    else:
        variable_lengths = False
    if 'embedding' in config.keys():
            embedding = config['embedding']
    else:
        embedding = None
    if 'update_embedding' in config.keys():
            update_embedding = config['update_embedding']
    else:
        update_embedding = True
    if 'decode_function' in config.keys():
            decode_function = config['decode_function']
    else:
        decode_function = F.log_softmax

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
    output_path_train = 'datasets/{}/train.output_data'.format(data_name)
    output_path_val = 'datasets/{}/val.output_data'.format(data_name)
    output_path_test = 'datasets/{}/test.output_data'.format(data_name)

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
    encoder = encoder_dict[encoder_name](vocab_size=vocab_size, max_len=max_len, hidden_dim=hidden_dim,
                                         input_dropout_rate=input_dropout_rate, dropout_rate=dropout_rate,
                                         num_layers=num_layers, bidirectional=bidirectional, rnn_cell=rnn_cell,
                                         variable_lengths=variable_lengths, embedding=embedding,
                                         update_embedding=update_embedding)
    decoder = decoder_dict[decoder_name](vocab_size=vocab_size, max_len=max_len, hidden_dim=hidden_dim,
                                         sos_id=sos_id, eos_id=eos_id, num_layers=num_layers, rnn_cell=rnn_cell,
                                         bidirectional=bidirectional, input_dropout_rate=input_dropout_rate,
                                         dropout_rate=dropout_rate)
    model = dl_model_dict[model_name](enccoder=encoder, decoder=decoder, decode_function=decode_function)

    # 根据不同特征选择调用的数据生成器类型
    if feature == 'bert':
        generator = data_loader.bert_data_generator
    elif feature == 'hierarchical':
        generator = data_loader.hierarchical_data_generator
    else:
        generator = data_loader.data_generator

    # 调用这个类的train_model函数来训练这个模型。
    model.train_model(model, generator, input_path_train, output_path_train, word_dict,
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
    # config_path = './config/aapr/dl/aapr.dl.seq2seq.norm.json'
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
    if model_name in dl_model_dict:
        # 运行深度学习的main函数。
        main_dl(config)
    else:
        raise RuntimeError("There is no model name.".format(model_name))

    # 程序结束时的时间。
    end_time = datetime.datetime.now()
    # 计算程序总共运行的时间，单位是秒（s）。
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))
    print('Done main!')
