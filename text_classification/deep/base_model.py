#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Base_Model
======
A class for something.
"""

import argparse
import datetime
import torch
import torch.nn as nn
import numpy as np
from utils.metrics import cal_all

class BaseModel(nn.Module):
    # 超参数->人类，参数->AI
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(BaseModel, self).__init__()
        # 一些重要的参数，这些参数几乎在所有的深度学习模型中都会被用到。

        self.vocab_size = vocab_size  # 词表大小
        self.embed_dim = embed_dim  # 词嵌入/向量的维度
        self.hidden_dim = hidden_dim  # 隐藏层的维度
        self.num_classes = num_classes  # 类别数量
        self.dropout_rate = dropout_rate  # dropout的比率，即丢弃神经网络中的多少神经元（向量中的元素）
        self.learning_rate = learning_rate  # 优化器中的学习率
        self.num_epochs = num_epochs  # 要将数据迭代训练多少轮
        self.batch_size = batch_size  # 每次训练要喂给（feed）模型多少样本
        self.criterion_name = criterion_name  # 损失函数
        self.optimizer_name = optimizer_name  # 优化器

        self.metrics_num = 4
        if 'metrics_num' in kwargs:
            self.metrics_num = kwargs['metrics_num']

        # 类，对象
        # look-up table = [vocab_size, embed_dim]
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, _weight=None)
        # weights matrix = [embed_dim, hidden_dim]
        self.fc = nn.Linear(embed_dim, hidden_dim)
        # weights matrix =  [hidden_dim, num_classes] = [64, 2]
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.drop_out = nn.Dropout(dropout_rate)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.criterion_dict = {
            'NLLLoss': torch.nn.NLLLoss,
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss  # with softmax
        }
        self.optimizer_dict = {
            'Adam': torch.optim.Adam
        }

        if criterion_name not in self.criterion_dict:
            raise ValueError("There is no criterion_name: {}.".format(criterion_name))
        self.criterion = self.criterion_dict[criterion_name]()

        if optimizer_name not in self.optimizer_dict:
            raise ValueError("There is no optimizer_name: {}.".format(optimizer_name))
        self.optimizer = self.optimizer_dict[optimizer_name](self.parameters(), lr=self.learning_rate)

        self.gpu = gpu

        # 是否有GPU
        self.device = torch.device('cuda:{}'.format(self.gpu) if torch.cuda.is_available() else 'cpu')
        print("Device: {}.".format(self.device))

    # 模型构建，包括数据输入到输出的所有数学过程。
    # 此为深度学习的前向传播。
    def forward(self, x):
        # batch_x: [batch_size, seq_len]
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim] = [64, 100, 128]
        avg_embed = torch.mean(embed, dim=1)  # [batch_size, embed_dim]
        # out = self.softmax(self.fc(avg_embed))
        hidden = self.fc(avg_embed)  # [batch_size, hidden_dim]
        hidden = self.drop_out(hidden)  # [batch_size, hidden_dim]
        out = self.fc_out(hidden)  # [batch_size, num_classes]
        return out

    # 模型训练。
    def train_model(self, model, data_generator, input_path, output_path, word_dict,
                    input_path_val=None, output_path_val=None,
                    input_path_test=None, output_path_test=None,
                    save_folder=None):
        # 将模型对象传送到CPU或者GPU上。
        model.to(self.device)
        # 激活模型中的dropout等。

        best_score = 0
        # 将数据循环num_epochs轮来训练模型。
        for epoch in range(self.num_epochs):
            model.train()
            total_y, total_pred_label = [], []
            total_loss = 0
            step_num = 0
            sample_num = 0
            # 在每轮中，每次喂给（feed）模型batch_size个样本。
            for x, y in data_generator(input_path, output_path, word_dict, batch_size=self.batch_size):
                # 将原始数据转化为torch.LongTensor格式。
                # 并将输入和输出数据都传送到对应的设备上。
                batch_x = torch.LongTensor(x).to(self.device)
                batch_y = torch.LongTensor(y).to(self.device)
                # 用当前的模型生成预测结果，此处相当于调用了forward函数。
                # 模型的前向传播。
                batch_pred_y = model(batch_x)
                # 删除模型参数上累积的梯度。
                model.zero_grad()
                # 用预测值和真实值计算损失值（loss）
                loss = self.criterion(batch_pred_y, batch_y)
                # 给定loss值，计算参数的梯度。
                # 模型的反向传播。
                loss.backward()
                # 利用优化器更新参数。
                self.optimizer.step()

                # 从预测值进一步计算得到预测的标签。
                pred_y_label = list(np.argmax(batch_pred_y.cpu().detach().numpy(), axis=-1))

                # 将每个batch的真实值拼接在一起。
                total_y += list(y)
                # 将每个batch的预测值拼接在一起。
                total_pred_label += pred_y_label

                # 将loss加在一起。
                total_loss += loss.item() * len(y)
                step_num += 1
                sample_num += len(y)

            # 评价模型在训练数据集上的性能。
            print("Have trained {} steps.".format(step_num))
            metric = cal_all
            if self.metrics_num == 4:
                metric = cal_all
            metric_score = metric(np.array(total_y), np.array(total_pred_label))
            sorted_metric_score = sorted(metric_score.items(), key=lambda x: x[0])
            metrics_string = '\t'.join(['loss'] + [metric_name[1:] for metric_name, _ in sorted_metric_score])
            score_string = '\t'.join(['{:.2f}'.format(total_loss/sample_num)] + ['{:.2f}'.format(score) for _, score in sorted_metric_score])
            print("{}\t{}\t{}".format('train', epoch, metrics_string))
            print("{}\t{}\t{}".format('train', epoch, score_string))

            # 评价模型在验证数据集上的性能。
            if input_path_val and output_path_val:
                metric_score = \
                    self.eval_model(model, data_generator, input_path_val, output_path_val, word_dict, 'val', epoch)
                acc = metric_score['1acc']
                torch.save(model, '{}{}.ckpt'.format(save_folder, epoch))
                print("Save model to {}.".format('{}{}.ckpt'.format(save_folder, epoch)))
                # 以accuracy作为主要指标，将验证集上accuracy最大的轮次得到的模型保存下来。
                if acc > best_score:
                    best_score = acc
                    # 模型保存。
                    torch.save(model, '{}{}.ckpt'.format(save_folder, 'best'))
                    print("Save model to {}.".format('{}{}.ckpt'.format(save_folder, 'best')))

            # 评价模型在测试数据集上的性能。
            if input_path_test and output_path_test:
                self.eval_model(model, data_generator, input_path_test, output_path_test, word_dict, 'test', epoch)

        # 用最优的模型评价模型在测试数据集上的性能。
        if input_path_test and output_path_test:
            # 模型加载。
            model = torch.load('{}{}.ckpt'.format(save_folder, 'best'))
            print(model)
            model.eval()
            self.eval_model(model, data_generator, input_path_test, output_path_test, word_dict, 'test', 'final')

    # 模型验证。
    def eval_model(self, model, data_generator, input_path, output_path, word_dict, phase, epoch):
        # 将模型对象传递到对应的设备上。
        model.to(self.device)
        # 将dropout等失活。
        model.eval()
        total_y, total_pred_label = [], []
        total_loss = 0
        step_num = 0
        sample_num = 0
        # 此处没有epoch，因为只要遍历一轮数据即可得到测试结果，并不需要通过这个过程修改模型中的参数。
        # 按照batch的大小（size）依次选取训练数据和验证数据。
        for x, y in data_generator(input_path, output_path, word_dict, batch_size=self.batch_size):
            # 将原始数据转化为torch.LongTensor格式。
            # 并将输入和输出数据都传送到对应的设备上。
            batch_x = torch.LongTensor(x).to(self.device)
            batch_y = torch.LongTensor(y).to(self.device)
            # 给定输入值，用模型计算得到预测值。
            # 模型的前向传播。
            batch_pred_y = model(batch_x)
            # 计算损失值。在验证和测试阶段，此值仅用来参考，并不会据此修改模型参数。
            # 在验证和测试阶段，没有反向传播。
            loss = self.criterion(batch_pred_y, batch_y)
            # 从预测值进一步得到label。
            pred_y_label = list(np.argmax(batch_pred_y.cpu().detach().numpy(), axis=-1))

            # 将每个batch的真实值拼接在一起。
            total_y += list(y)
            # 将每个batch的预测值拼接在一起。
            total_pred_label += pred_y_label

            total_loss += loss.item() * len(y)
            step_num += 1
            sample_num += len(y)
        print("Have {} {} steps.".format(phase, step_num))
        metric = cal_all
        if self.metrics_num == 4:
            metric = cal_all
        metric_score = metric(np.array(total_y), np.array(total_pred_label))
        sorted_metric_score = sorted(metric_score.items(), key=lambda x: x[0])
        metrics_string = '\t'.join(['loss'] + [metric_name[1:] for metric_name, _ in sorted_metric_score])
        score_string = '\t'.join(
            ['{:.2f}'.format(total_loss / sample_num)] + ['{:.2f}'.format(score) for _, score in sorted_metric_score])
        print("{}\t{}\t{}".format(phase, epoch, metrics_string))
        print("{}\t{}\t{}".format(phase, epoch, score_string))
        return metric_score


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

    print('Done Base_Model!')
