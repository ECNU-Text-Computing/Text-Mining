import argparse
import datetime
import torch
import torch.nn as nn
from base_model import BaseModel


# RNN、多层RNN与双向RNN
class TextRNN(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        # 继承父类BaseModel的属性
        super(TextRNN, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                      dropout_rate, learning_rate, num_epochs, batch_size,
                                      criterion_name, optimizer_name, gpu, **kwargs)

        # RNN堆叠的层数，默认为2层
        self.num_layers = 2
        if 'num_layers' in kwargs:
            self.num_layers = kwargs['num_layers']
        # RNN的方向性，双向RNN则取值2；单向RNN则取1
        self.num_directions = 2
        if 'num_directions' in kwargs:
            self.num_directions = kwargs['num_directions']
        self.bidirection = True if self.num_directions == 2 else False

        # 设置RNN模型的参数
        self.rnn = nn.RNN(embed_dim, hidden_dim, self.num_layers, batch_first=True, dropout=dropout_rate,
                            bidirectional=self.bidirection)
        # 设置输出层的参数
        self.fc_out = nn.Linear(hidden_dim * self.num_directions, num_classes)

    # 模型的前向传播
    def forward(self, x):
        # batch_x: [batch_size, seq_len]
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        hidden, _ = self.rnn(embed)  # [batch_size, seq_len, hidden_dim * self.num_directions]
        # -1表示取该维度从后往前的第一个，即只需要最后一个词的隐藏状态作为输出
        hidden = self.dropout(hidden[:, -1, :])  # [batch_size, hidden_dim * self.num_directions]
        out = self.fc_out(hidden)  # [batch_size, num_classes]
        return out


# 程序入口
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print("This is a test process.")

        # 设置测试用例，验证模型是否能够运行
        # 设置模型参数
        vocab_size, embed_dim, hidden_dim, num_classes, \
        dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu = 100, 64, 32, 2, 0.5, 0.001, 3, 32, 'CrossEntropyLoss', 'Adam', 0
        # 创建类的实例
        model = TextRNN(vocab_size, embed_dim, hidden_dim, num_classes,
                        dropout_rate, learning_rate, num_epochs, batch_size,
                        criterion_name, optimizer_name, gpu)
        # 传入简单数据，查看模型运行结果
        input_data = torch.LongTensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [1, 4, 2, 7, 5]])
        output_data = model(input_data)
        print("The output is: {}".format(output_data))

        print("The test process is done.")

    else:
        print("There is no {} function. Please check your command.".format(args.phase))

    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Text_RNN.')
