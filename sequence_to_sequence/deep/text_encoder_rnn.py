import argparse
import datetime
import torch.nn as nn
from .base_rnn import BaseRNN


# RNN、多层RNN与双向RNN
class EncoderRNN(BaseRNN):
    def __init__(self, vocab_size, max_len, hidden_dim,
                 input_dropout_rate=0, dropout_rate=0,
                 num_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False,
                 embedding=None, update_embedding=True):
        # 继承父类BaseModel的属性
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_dim, input_dropout_rate, dropout_rate,
                                         num_layers, rnn_cell)
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(hidden_dim, hidden_dim, num_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_rate)

    # 模型的前向传播
    def forward(self, input_var, input_lengths=None):
        embed = self.embedding(input_var)
        embed = self.input_dropout(embed)
        if self.variable_lengths:
            embed = nn.utils.rnn.pack_padded_sequence(embed, input_lengths, batch_first=True)
        output, hidden = self.rnn(embed)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


# 程序入口
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print("This is a test process.")

    else:
        print("There is no {} function. Please check your command.".format(args.phase))

    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Encoder_RNN.')
