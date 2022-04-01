#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Base_Model
======
A class for something.
"""
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from sequence_labeling.data_loader import DataLoader
from sequence_labeling.data_processor import DataProcessor
from sequence_labeling.utils.evaluate import Evaluator

torch.manual_seed(1)


class BaseModel(nn.Module):

    def __init__(self, data_root, model_save_path, embedding_dim, hidden_dim, tag_size, vocab_size,
                 num_epochs, batch_size, learning_rate, criterion_name, optimizer_name, **kwargs):
        super(BaseModel, self).__init__()
        self.data_root = data_root
        self.model_save_path = model_save_path

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tag_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.criterion_name = criterion_name
        self.optimizer_name = optimizer_name
        self.criterion_dict = {
            'NLLLoss': torch.nn.NLLLoss,
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss
        }
        self.optimizer_dict = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam
        }

        self.word_embeddings = nn.Embedding(self.vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def run_Model(self, model, op_mode):
        if self.criterion_name not in self.criterion_dict:
            raise ValueError("There is no criterion_name: {}.".format(self.criterion_name))
        loss_function = self.criterion_dict[self.criterion_name]()
        if self.optimizer_name not in self.optimizer_dict:
            raise ValueError("There is no optimizer_name: {}.".format(self.optimizer_name))
        optimizer = self.optimizer_dict[self.optimizer_name](model.parameters(), lr=self.learning_rate)

        if op_mode == 'train':
            model.train()
            for epoch in range(self.num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
                for sentenceList, tagList in DataLoader().data_generator(batch_size=self.batch_size, op_mode=op_mode):
                    for i in range(len(sentenceList)):
                        batch_x = torch.LongTensor(sentenceList[i])
                        batch_y = torch.LongTensor(tagList[i])
                        model.zero_grad()
                        tag_scores = model(batch_x)
                        loss = loss_function(tag_scores, batch_y)
                        loss.backward()
                        optimizer.step()

            model_save_path = self.data_root + self.model_save_path
            torch.save(model, '{}'.format(model_save_path))
        elif op_mode == 'eval' or 'test':
            model.eval()
        else:
            print("op_mode参数未赋值(train/eval/test)")

        with torch.no_grad():
            for sentenceList, tagList in DataLoader().data_generator(op_mode=op_mode):
                batch_x = torch.LongTensor(sentenceList[0])
                print(batch_x)
                scores = model(batch_x)

                # print("After Train:", scores)
                # print('标注结果转换为Tag索引序列：', torch.max(scores, dim=1))
                y_predict = list(torch.max(scores, dim=1)[1].numpy())

                # 将索引序列转换为Tag序列
                y_pred = self.index_to_tag(y_predict)
                y_true = self.index_to_tag(tagList[0])

                # 输出评价结果
                print(y_pred)
                print(y_true)
                print(Evaluator().classifyreport(y_true, y_pred))


    def index_to_tag(self, y):
        tag_index_dict = DataProcessor().load_tags()
        index_tag_dict = dict(zip(tag_index_dict.values(), tag_index_dict.keys()))
        y_tagseq = []
        for i in range(len(y)):
            y_tagseq.append(index_tag_dict[y[i]])
        return y_tagseq


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
