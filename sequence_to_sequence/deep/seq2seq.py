import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description="Process some description.")
    parser.add_argument('--phase', default='test', help='the function name')
    args = parser.parse_args()

    if args.phase == 'test':
        print("This is a test process.")

        vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu, num_layers, num_directions \
            = 100, 64, 32, 2, 0.5, 0.001, 3, 32, 'CrossEntropyLoss', 'Adam', 0, 2, 2
        # 创建类的实例
        model = Seq2seq(vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, num_epochs,
                           batch_size, criterion_name, optimizer_name, gpu,
                           num_layers=num_layers, num_directions=num_directions)
        # 传入简单数据，查看模型运行结果
        # [batch_size, seq_len] = [3, 5]
        input_data = torch.LongTensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [1, 4, 2, 7, 5]])
        output_data = model(input_data)
        print("The output_data is: {}".format(output_data))

        print("The test process is done.")

    else:
        print("There is no {} function.".format(args.phase))

    end_time = datetime.datetime.now()
    print("{} takes {} seconds.".format(args.phase, (end_time - start_time).seconds))
    print("Done DecoderRNN.")
