import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from Deep.Base_Model import Base_Model
from utils.metrics import cal_all


class TextDPCNN(Base_Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_filters,dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(TextDPCNN,self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)

        self.num_filters=num_filters
        self.conv_region_embedding = nn.Conv2d(1, num_filters, (3,embed_dim),
                                                   stride=1)
        self.conv = nn.Conv2d(num_filters, num_filters, (3, 1), stride=1)
        # kernel_size是maxpooling的窗口大小，降维
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        #零填充函数（左，右，上，下）
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1)) #卷积前后的尺寸不变
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1)) #对池化进行填充
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(num_filters, 2)


    def forward(self, x):
       # print(x.size())
        x=self.embedding(x) #[batch_size,seq_len,embed_dim]
       # print(x.size())
        x = x.unsqueeze(1) #[batch_size,1,seq_len,embed_dim]
       # print(x.size())

        x = self.conv_region_embedding(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        # pad保证等长卷积，先通过激活函数再卷积
        x = self.padding_conv(x)  # [batch_size, num_filters, seq_len, 1]
        # print (x.size())
        x = self.act_fun(x)

        x = self.conv(x)   # [batch_size, num_filters, seq_len-3+1, 1]
        # print (x.size())
        x = self.padding_conv(x)   # [batch_size, num_filters, seq_len, 1]
        # print (x.size())
        x = self.act_fun(x)
        x = self.conv(x)   # [batch_size, num_filters, seq_len-3+1, 1]
        # print (x.size())

        while x.size()[-2] > 1:
            x = self._block(x)

        x=x.squeeze()

        x = self.linear_out(x)

        return x

    def _block(self, x):

        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)
        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv(x)
        # Short Cut
        x = x + px
        return x

    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels

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
