import argparse
import datetime
import torch
from deep.base_model import BaseModel
from torch import nn
import torch.nn.functional as F


class DPCNN(BaseModel):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(DPCNN, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                    dropout_rate, learning_rate, num_epochs, batch_size,
                                    criterion_name, optimizer_name, gpu, **kwargs)
        # 简单CNN参数设置
        # 卷积核数量
        self.num_filters = 4
        if 'num_filters' in kwargs:
            self.num_filters = kwargs['num_filters']
        # 卷积层和池化层设定
        self.conv_region = nn.Conv2d(1, self.num_filters, (3, self.embed_dim), stride=1)
        self.conv = nn.Conv2d(self.num_filters, self.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.relu = nn.ReLU()
        # 零填充函数（左，右，上，下）
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # 卷积前后的尺寸不变
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # 对池化进行填充
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.num_filters, self.num_classes)

    def forward(self, content):
        # 不等长段落进行张量转化
        x = self.embedding(content)  # [batch_size, seq_len, embed_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        x = self.conv_region(x)  # [batch_size, num_filters, seq_len-3+1, 1]

        # padding保证等长卷积，先通过激活函数再卷积
        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]

        while x.size()[2] >= 2:
            x = self._block(x) 
        # 循环结束后x的形状：[batch_size, num_filters, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [batch_size, num_filters]
        x = self.dropout(x)
        x = self.fc(x)  # [batch_size, num_classes]
        return x

    def _block(self, x):
        # 池化
        x = self.padding2(x)
        px = self.max_pool(x)

        # 卷积
        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        x = x + px
        return x


if __name__ == '__main__':
    if __name__ == '__main__':
        start_time = datetime.datetime.now()
        parser = argparse.ArgumentParser(description='Process some description.')
        parser.add_argument('--phase', default='test', help='the function name.')

        args = parser.parse_args()

        if args.phase == 'test':
            vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, \
            num_epochs, batch_size, criterion_name, optimizer_name, gpu \
                = 100, 64, 32, 2, 0.5, 0.0001, 3, 64, 'CrossEntropyLoss', 'Adam', 0

            model = DPCNN(vocab_size, embed_dim, hidden_dim, num_classes,
                          dropout_rate, learning_rate, num_epochs, batch_size,
                          criterion_name, optimizer_name, gpu)

            input = torch.LongTensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
                                      [1, 3, 5, 7, 9], [2, 4, 6, 8, 10],
                                      [1, 4, 8, 3, 6]])  # [batch_size, seq_len] = [6, 5]
            output = model(input)
            print(output)

            print('The test process is done.')
        else:
            print("There is no {} function. Please check your command.".format(args.phase))
        end_time = datetime.datetime.now()
        print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

        print('Done DPCNN!')

