import argparse
import datetime
import torch
from text_cnn import TextCNN
from torch import nn
import torch.nn.functional as F


class CNNAttention(TextCNN):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(CNNAttention, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                           dropout_rate, learning_rate, num_epochs, batch_size,
                                           criterion_name, optimizer_name, gpu, **kwargs)
        self.model_name = 'CNNAttention'
        # 以点积方式计算注意力
        self.out_trans = nn.Linear(self.num_filters, self.num_filters)
        self.w = nn.Linear(self.num_filters, 1, bias=False)
        self.fc = nn.Linear(self.num_filters, self.num_classes)

    def forward(self, x):
        # input: [batch_size, seq_len]
        out = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        # 增加维度。由于卷积操作是在二维平面上进行的，而词向量内部不能拆分（拆分没有意义），因此需要增加一维
        out = out.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        # 进行卷积和池化操作，并将结果按列（即维数1）拼接
        # [batch_size, seq_len - len(filter_sizes) +1, num_filters]
        out = torch.cat([self.con_and_pool(out, conv).unsqueeze(1) for conv in self.convs], dim=1)
        # 计算注意力
        M = F.tanh(self.out_trans(out))  # [batch_size, seq_len - len(filter_sizes) +1, num_filters]
        alpha = F.softmax(self.w(M), dim=1)  # [batch_size, seq_len - len(filter_sizes) +1, 1]
        out = torch.sum(out * alpha, dim=1)  # [batch_size, num_filters]
        # 上面一行也可写为：out = torch.matmul(out * alpha)
        # 通过dropout和全连接层后输出
        out = self.drop_out(out)  # [batch_size, num_filters]
        out = self.fc(out)  # [batch_size, num_classes]
        return out


if __name__ == '__main__':
    if __name__ == '__main__':
        start_time = datetime.datetime.now()
        parser = argparse.ArgumentParser(description='Process some description.')
        parser.add_argument('--phase', default='test', help='the function name.')

        args = parser.parse_args()

        if args.phase == 'test':
            # 设置测试用例，验证模型是否能够运行
            # 设置模型参数
            vocab_size, embed_dim, hidden_dim, num_classes, \
            dropout_rate, learning_rate, num_epochs, batch_size, \
            criterion_name, optimizer_name, gpu, num_filters, filter_sizes \
                = 100, 64, 32, 2, 0.5, 0.001, 3, 32, 'CrossEntropyLoss', 'Adam', 0, 2, [1, 2, 3]
            # 创建类的实例
            model = CNNAttention(vocab_size, embed_dim, hidden_dim, num_classes,
                            dropout_rate, learning_rate, num_epochs, batch_size,
                            criterion_name, optimizer_name, gpu, num_filters=num_filters, filter_sizes=filter_sizes)
            # 传入简单数据，查看模型运行结果
            # input_data: [batch_size, seq_len] = [3, 5]
            input_data = torch.LongTensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 4, 2, 7, 5]])
            output_data = model(input_data)
            print("The output is: {}".format(output_data))

            print("The test process is done.")
        else:
            print('error! No such method!')
        end_time = datetime.datetime.now()
        print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

        print('Done CNN Attention!')
