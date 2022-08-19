import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import datetime
from base_model import BaseModel


class Attention(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(Attention, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                        dropout_rate, learning_rate, num_epochs, batch_size,
                                        criterion_name, optimizer_name, gpu, **kwargs)
        self.tanh = nn.Tanh()
        self.out_trans = nn.Linear(self.embed_dim, self.embed_dim)
        # 设置以内积方式取得注意力的权重矩阵
        self.w = nn.Linear(self.embed_dim, 1, bias=False)

    def forward(self, x):
        # input: [batch_size, seq_len]
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        M = self.tanh(self.out_trans(embed))  # [batch_size, seq_len, embed_dim]
        alpha = F.softmax(self.w(M), dim=1)  # 注意：进行softmax操作时一定要指定维度
        out = embed * alpha  # [batch_size, seq_len, embed_dim]
        print('out的形状是：{}'.format(out.size()))
        out = torch.sum(out, dim=1)  # [batch_size, embed_dim]
        print('out2的形状是：{}'.format(out.size()))
        out = F.relu(out)
        # 通过两个全连接层后输出
        out = self.fc1(out)
        out = self.fc_out(out)
        return out


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, \
        num_epochs, batch_size, criterion_name, optimizer_name, gpu \
            = 100, 64, 32, 2, 0.5, 0.0001, 3, 64, 'CrossEntropyLoss', 'Adam', 0

        model = Attention(vocab_size, embed_dim, hidden_dim, num_classes,
                          dropout_rate, learning_rate, num_epochs, batch_size,
                          criterion_name, optimizer_name, gpu)

        input = torch.LongTensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
                                  [1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 4, 8, 3, 6]])  # [batch_size, seq_len] = [6, 5]
        output = model(input)
        print(output)

        print('The test process is done.')

    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Attention!')
