import argparse
import datetime
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from deep.text_rnn import TextRNN


# RNN、多层RNN与双向RNN
class EncoderRNN(TextRNN):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        # 继承父类BaseModel的属性
        super(EncoderRNN, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                         dropout_rate, learning_rate, num_epochs, batch_size,
                                         criterion_name, optimizer_name, gpu, **kwargs)

        # 设置输出层的参数
        self.fc_out = nn.Linear(self.hidden_dim * self.num_directions, self.num_classes)

    # 模型的前向传播
    def forward(self, x):
        # input: [batch_size, seq_len]
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        seq_len = [s.size(0) for s in embed]
        embed = pack_padded_sequence(embed, seq_len, batch_first=True, enforce_sorted=False)
        output, hidden = self.model(embed)  # [batch_size, seq_len, hidden_dim * num_directions]
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output, hidden


# 程序入口
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print("This is a test process.")

        # 设置测试用例，验证模型是否能够运行
        # 设置模型参数，测试所用模型为2层的双向LSTM
        vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu, num_layers, num_directions \
            = 100, 64, 32, 2, 0.5, 0.001, 3, 32, 'CrossEntropyLoss', 'Adam', 0, 2, 2

        model = EncoderRNN(vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, num_epochs,
                           batch_size, criterion_name, optimizer_name, gpu,
                           num_layers=num_layers, num_directions=num_directions)
        # 传入简单数据，查看模型运行结果
        # input_data: [batch_size, seq_len] = [3, 5]
        input_data = torch.LongTensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [1, 4, 2, 7, 5]])
        output_data = model(input_data)
        print("The output_data is: {}".format(output_data))

        print("The test process is done.")

    else:
        print("There is no {} function. Please check your command.".format(args.phase))

    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Encoder_RNN.')