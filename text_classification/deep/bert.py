#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
BERT(本模型尚存在问题，待后续修改)
======
参考文献：
    1. https://blog.csdn.net/weixin_42237487/article/details/112355703
    2. https://huggingface.co/docs/transformers/v4.21.1/en/model_doc/bert#transformers.BertModel
BERT特点：
    1. BERT可以解决词的情态表征，解决一词多义问题。
        Bert生成动态Word Embedding的思路：
        事先用一个学到的单词的Word Embedding，该Word Embedding在使用的时候已经具备了特定的上下文意思了，
        可以根据上下文单词的语义去调整（自注意力）单词的Word Embedding表示，
        经过调整后的Word Embedding更能表达在这个上下文中的具体含义，自然也就解决了多义词的问题了。
    2. Bert编码器（特征提取）最终可以输出两个Embedding信息：单词的Word Embedding和句子的Embedding。
    3. 可以提取出Bert编码器输出的句子Embedding，经过全连接的网络层对句子进行情感判断。这个Embedding携带了当前句子的主要信息。
模型由输入层、编码层和输出层三个部分组成
BERT的输入：input_data
    1. Token Embedding：词特征（词向量）的嵌入，针对中文，目前只支持字特征嵌入
    2. Segment Embedding：词的句子级特征嵌入，针对双句子输入任务，做句子A，B嵌入，针对单句子任务，只做句子A嵌入
    3. Position Embedding：词的位置特征，针对中文，目前最大长度为 512
BERT的输出：
    1. last_hidden_state：torch.FloatTensor类型，最后一个隐藏层的序列的输出，形状为[batch_size, seq_len, hidden_size]
    2. pooler_output：torch.FloatTensor类型，[CLS]的这个token的输出，形状为[batch_size, hidden_size]
    3. hidden_states(可选项)：tuple(torch.FloatTensor)类型。需要指定config.output_hidden_states=True。
    它是一个元组，第一个元素是embedding，其余元素是各层的输出，每个元素的形状为[batch_size, seq_len, hidden_size]
    4. attentions(可选项)：tuple(torch.FloatTensor)类型。需要指定config.output_attentions=True。
    它也是一个元组，它的元素是每一层的注意力权重，用于计算self-attention heads的加权平均值，形状为[batch_size, seq_len, hidden_size]
BERT的下游任务：MLM或NSP
    1、MLM：掩码语言模型
    2、NSP：句子连贯性判断
======
[UNK]：文本中的元素不在词典中，用该符号表示生僻字。对应编码是100
[CLS]:用于分类场景，该位置可表示整句话的语义。对应的编码是101
[SEP]: 表示分隔开两个句子。对应的编码是102
[MASK] ：表示这个词被遮挡。需要带着[]，并且mask是大写，对应的编码是103
[PAD]：针对有长度要求的场景，填充文本长度，使得文本长度达到要求。对应编码是0
"""

import argparse
import datetime
import random
import torch
import torch.nn as nn
import numpy as np
from utils.metrics import cal_all
from transformers import BertModel, BertTokenizer


# BERT文本分类
class BERT(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(BERT, self).__init__()
        # 一些重要的参数，这些参数几乎在所有的深度学习模型中都会被用到。

        self.vocab_size = vocab_size  # 词表大小
        self.embed_dim = embed_dim  # 词嵌入/向量的维度
        self.hidden_dim = 768  # 隐藏层的维度，bert模型的输出向量为768维
        self.num_classes = num_classes  # 类别数量
        self.dropout_rate = dropout_rate  # dropout的比率，即丢弃神经网络中的多少神经元（向量中的元素）
        self.learning_rate = learning_rate  # 优化器中的学习率
        self.num_epochs = num_epochs  # 要将数据迭代训练多少轮
        self.batch_size = batch_size  # 每次训练要喂给（feed）模型多少样本
        self.criterion_name = criterion_name  # 损失函数
        self.optimizer_name = optimizer_name  # 优化器

        # 加载的预训练模型名称
        self.model_path = 'bert-base-uncased'
        if 'model_path' in kwargs:
            self.model_path = kwargs['model_path']
        # 隐藏层维度设置
        self.hidden_dim2 = self.hidden_dim // 2
        if 'hidden_dim2' in kwargs:
            self.hidden_dim2 = kwargs['hidden_dim2']
        # 设置模型评价指标数量
        self.metrics_num = 4
        if 'metrics_num' in kwargs:
            self.metrics_num = kwargs['metrics_num']

        # 类，对象
        # look-up table = [vocab_size, embed_dim]
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, _weight=None)
        # bert模型设置
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.bert = BertModel.from_pretrained(self.model_path, return_dict=True, output_attentions=True,
                                              output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad = True
        # 根据输出层的不同复杂程度，设置不同模式
        # 默认为普通模式，仅通过一个全连接层
        self.mode = 'normal'
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        if 'mode' in kwargs:
            if kwargs['mode'] == 'adap':
                # 进阶模式，通过两个全连接层后输出
                print('Using adap mode.')
                self.mode = 'adap'
                self.fc = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim2),
                    nn.Tanh(),
                    nn.Linear(self.hidden_dim2, self.num_classes)
                )
            elif kwargs['mode'] == 'pro':
                # 高级模式，通过三个全连接层后输出
                print('Using pro mode.')
                self.fc = nn.Sequential(
                    nn.Linear(self.hidden_dim, int(self.hidden_dim / 2)),
                    # nn.ReLU(),
                    nn.Tanh(),
                    nn.Dropout(p=self.dropout_rate),
                    nn.Linear(int(self.hidden_dim / 2), int(self.hidden_dim / 4)),
                    # nn.ReLU(),
                    nn.Tanh(),
                    nn.Dropout(p=self.dropout_rate),
                    nn.Linear(int(self.hidden_dim / 4), self.num_classes)
                )
        else:
            print('Using normal mode.')
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

    def forward(self, x):
        # input: [batch_size, seq_len]
        output = self.bert(x)['last_hidden_state']
        out = torch.mean(output, dim=1)  # [batch_size, hidden_dim]
        out = self.drop_out(out)  # [batch_size, num_classes]
        out = self.fc(out)  # [batch_size, num_classes]

        # 用from pytorch_pretrained_bert时的输出
        # _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # output_all_encoded_layers=True时，12层Transformer的结果全返回，存在第一个列表中，
        # 每个encoder_output的大小为[batch_size, sequence_length, hidden_dim]
        # pool_out = [batch_size, hidden_dim]，取了最后一层Transformer的输出结果的第一个单词[cls]的hidden states，其已经蕴含了整个input句子的信息了
        # out = self.fc(pooled)

        return out

    def data_generator(self, input_path, output_path, batch_size, shuffle=True):

        print("Load input_data data from {}.".format(input_path))
        print("Load output_data from {}.".format(output_path))

        # 读取输入数据。
        # type(src) = list
        with open(input_path, 'r') as fp:
            input_data = fp.readlines()

        # 读取输出数据。
        # type(out) = list
        with open(output_path, 'r') as fp:
            output_data = fp.readlines()

        # 如果你的机器资源不够，一次跑不完数据，为了验证整个流程（pipeline）是对的，可以采用小一点的数据规模。
        # 比方说，这里设置了1000，你可以可以设置100或者200等。
        # 如果你可以跑全量数据，那么你可以注释掉这行代码。
        # 记得有一些可用可不用的代码，可以注释掉而不要直接删掉。免得以后用的时候还要重新写。
        # src = src[:1000]
        # out = out[:1000]

        # 是否将数据随机打乱？
        if shuffle:
            data = list(zip(input_data, output_data))
            random.shuffle(data)
            input_data, output_data = zip(*data)

        # 按照batch将数据输出给深度学习模型。
        # 注意，因为下面使用了yield而非return，所以这个函数是一个生成器。
        for i in range(0, len(input_data), batch_size):
            batch_x = self.tokenizer(input_data[i: i + batch_size], padding=True)['input_ids']

            # 提取输出数据的下一个batch。
            batch_output = [int(label.strip()) for label in output_data[i: i+batch_size]]
            batch_y = np.array(batch_output)

            yield batch_x, batch_y

    def train_model(self, model, input_path, output_path,
                    input_path_val=None, output_path_val=None,
                    input_path_test=None, output_path_test=None,
                    save_folder=None):
        # 将模型对象传送到CPU或者GPU上。
        model.to(self.device)

        data_generator = self.data_generator(input_path, output_path, batch_size=self.batch_size)

        best_score = 0
        # 将数据循环num_epochs轮来训练模型。
        for epoch in range(self.num_epochs):
            model.train()
            total_y, total_pred_label = [], []
            total_loss = 0
            step_num = 0
            sample_num = 0
            # 在每轮中，每次喂给（feed）模型batch_size个样本。
            for x, y in data_generator:
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
                    self.eval_model(model, input_path_val, output_path_val,'val', epoch)
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
                self.eval_model(model, input_path_test, output_path_test, 'test', epoch)

        # 用最优的模型评价模型在测试数据集上的性能。
        if input_path_test and output_path_test:
            # 模型加载。
            model = torch.load('{}{}.ckpt'.format(save_folder, 'best'))
            print(model)
            model.eval()
            self.eval_model(model, input_path_test, output_path_test, 'test', 'final')

    # 模型验证。
    def eval_model(self, model, input_path, output_path, phase, epoch):
        data_generator = self.data_generator(input_path, output_path, batch_size=self.batch_size)
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
        for x, y in data_generator:
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
        vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu, mode \
            = 100, 64, 32, 2, 0.5, 0.0001, 3, 8, 'CrossEntropyLoss', 'Adam', 0, 'pro'

        # 测试所用为pro模式
        model = BERT(vocab_size, embed_dim, hidden_dim, num_classes,
                     dropout_rate, learning_rate, num_epochs, batch_size,
                     criterion_name, optimizer_name, gpu, mode=mode)
        # 测试数据为id时
        '''
        input_data = torch.LongTensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
                                       [1, 3, 5, 7, 9], [2, 4, 6, 8, 10],
                                       [1, 4, 8, 3, 6]])  # [batch_size, seq_len] = [6, 5]
        output_data = model(input_data)
        '''
        # 测试数据为文本时
        with open(r'E:\GitRepo\Text-Mining\text_classification\deep\test_data.txt', 'r') as fp:
            input_data = fp.readlines()
        output_data = model(input_data)
        print(output_data)
        print('The test process is done!')

    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done BERT Model!')
