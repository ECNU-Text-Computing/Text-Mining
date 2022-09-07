import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from text_cnn import TextCNN


class CNNLSTM(TextCNN):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        # 继承父类TextRNN的属性
        super(CNNLSTM, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                      dropout_rate, learning_rate, num_epochs, batch_size,
                                      criterion_name, optimizer_name, gpu, **kwargs)

        # RNN堆叠的层数，默认为1层
        self.num_layers = 1
        if 'num_layers' in kwargs:
            self.num_layers = kwargs['num_layers']
        # RNN的方向性，双向RNN则取值2；单向RNN则取1
        self.num_directions = 1
        if 'num_directions' in kwargs:
            self.num_directions = kwargs['num_directions']
        self.bidirectional = True if self.num_directions == 2 else False

        # 使用的神经网络类型，默认为LSTM
        self.rnn_model = 'LSTM'
        if 'rnn_model' in kwargs:
            self.rnn_model = kwargs['rnn_model']

        # RNN模型初始化
        in_dim = self.out_channels * len(self.filter_sizes)
        if self.rnn_model == 'LSTM':
            self.model_name = 'LSTMAttention'
            if self.bidirectional:
                self.model_name = 'BiLSTMAttention'
            self.model = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=self.num_layers,
                                 dropout=dropout_rate, batch_first=True, bidirectional=self.bidirectional)
        elif self.rnn_model == 'GRU':
            self.model_name = 'GRUAttention'
            if self.bidirectional:
                self.model_name = 'BiGRUAttention'
            self.model = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, num_layers=self.num_layers,
                                dropout=dropout_rate, batch_first=True, bidirectional=self.bidirectional)
        elif self.rnn_model == 'RNN':
            self.model_name = 'RNNAttention'
            if self.bidirectional:
                self.model_name = 'BiRNNAttention'
            self.model = nn.RNN(input_size=in_dim, hidden_size=hidden_dim, num_layers=self.num_layers,
                                dropout=dropout_rate, batch_first=True, bidirectional=self.bidirectional)
        else:
            print('No such RNN model!')

        # 设置输出层的参数
        self.fc_out = nn.Linear(self.hidden_dim * self.num_directions, self.num_classes)

    # 模型的前向传播
    def forward(self, x):
        # input: [batch_size, seq_len]
        batch_size = x.size(0)
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embed = embed.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        # 进行卷积操作，提取句子特征
        cnn_out = [F.relu(conv(embed)).squeeze().reshape(batch_size * self.out_channels, -1) for conv in self.convs]
        # 手动padding
        rnn_in = []
        for i in cnn_out:
            rnn_in.append(i.permute(1, 0))
        rnn_in = pad_sequence(rnn_in).permute(0, 2, 1)\
            .reshape(batch_size, self.out_channels * len(self.filter_sizes), -1)
        rnn_in = rnn_in.permute(0, 2, 1)  # [batch_size, seq_len-min(filter_sizes)+1, out_channels*len(filter_sizes)]
        # 将pad过的数据输入RNN模型
        hidden, _ = self.model(rnn_in)  # [batch_size, seq_len - min(filter_sizes) + 1, hidden_dim * num_directions]
        hidden = torch.mean(hidden, dim=1)  # [batch_size, hidden_dim * num_directions]
        hidden = self.drop_out(hidden)  # [batch_size, hidden_dim * num_directions]
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
        # 设置模型参数，测试所用模型为3卷积核2通道的CNN+2层双向LSTM
        vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu, num_layers, num_directions, out_channels, filter_sizes \
            = 100, 64, 32, 2, 0.5, 0.001, 3, 32, 'CrossEntropyLoss', 'Adam', 0, 2, 2, 2, [2, 3, 4]
        # 创建类的实例
        model = CNNLSTM(vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, num_epochs,
                        batch_size, criterion_name, optimizer_name, gpu, num_layers=num_layers,
                        num_directions=num_directions, out_channels=out_channels, filter_sizes=filter_sizes)
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