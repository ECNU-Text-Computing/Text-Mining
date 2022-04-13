import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from Deep.Base_Model import Base_Model
from sklearn.feature_extraction.text import TfidfVectorizer
from Data_Loader import Data_Loader


class TextRCNN(Base_Model):

    """配置参数"""
    def __init__(self,vocab_size, embed_dim,filter_sizes, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, num_layers,require_improvement,pad_size,gpu):
       super(TextRCNN, self).__init__(vocab_size, embed_dim,filter_sizes, hidden_dim, num_classes,
                                  dropout_rate, learning_rate, num_epochs, batch_size,
                                  criterion_name, optimizer_name, num_layers,require_improvement,pad_size,gpu)
       self.filter_sizes=filter_sizes
       self.num_layers=num_layers
       self.lstm = nn.LSTM(embed_dim,hidden_dim, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout_rate)
       self.maxpool = nn.MaxPool1d(pad_size)
       self.fc = nn.Linear(hidden_dim * 2, num_classes)
        

    def forward(self, x):
        # Text RCNN: input -> embedding -> RNN -> Max Pooling -> output
        # NOTE: You should check other details of the original TextRCNN

        # input x: [batch_size, seq_len] = [3, 5]

        # embed: [batch_size, seq_len, embedding] = [3, 5, 64]
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # Please refer to API of LSTM. Especially, you should know the shape of input and output.
        # Here, the shapes are as follows:
        # out: [batch_size, seq_len, 2*hidden_dim] = [3, 5, 128], as you set the bidirectional as True.
        # hidden_state: [2*num_layers, hidden_dim]
        # cell_state: [2*num_layers, hidden_dim]
        
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, (hidden_state,cell_state)= self.lstm(embed)
        print(out.size())

        # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        # Please refer to API of MaxPool2d.
        # Here, the size of pooling kernel is [seq_len, 1]. We use out.size(1) to get the seq_len.
        # pooled: [batch_size, 1, 2*hidden_dim] = [3, 1, 128]
        pooled = nn.MaxPool2d((out.size(1), 1))(out)
        print(pooled.size())

        # We should remove the invalid dimension. In that case, it is the second dimension.
        # Since the dimensions start at 0, we remove the 1-dimension.
        # pooled: [batch_size, 2*hidden_dim] = [3, 128]
        pooled = pooled.squeeze(1)
        print(pooled.size())
        
        # Finally, we use a fully connected layer to get output labels.
        pred = self.fc(pooled)
        print(pred.size())
        
        return pred

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    # python3 text_rcnn.py --phase test
    if args.phase == 'test':
        # For testing our model, we can set the hyper-parameters as follows.
        vocab_size, embed_dim, hidden_dim, num_classes, \
        num_layers, \
        dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu = \
            200, 64, 64, 2, 2, 0.5, 0.0001, 1, 64, 'CrossEntropyLoss', 'Adam', 1

        # new an objective.
        model = TextRCNN(vocab_size, embed_dim, hidden_dim, num_classes,
                         num_layers,
                         dropout_rate, learning_rate, num_epochs, batch_size,
                         criterion_name, optimizer_name, gpu)

        # a simple example of the input.
        input = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
        input = torch.LongTensor(input)  # input: [batch_size, seq_len] = [3, 5]

        # the designed model can produce an output.
        pred = model(input)
        print(pred)
        
        print('This is a test process.')
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Base_Model!')
