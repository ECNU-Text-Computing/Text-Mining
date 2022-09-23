import datetime
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
from deep.text_rnn import TextRNN
import torch.nn.functional as F


class DecoderRNN(TextRNN):
    KEY_ATTN_SCORE = 'attention_score' 
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(DecoderRNN, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                         dropout_rate, learning_rate, num_epochs, batch_size,
                                         criterion_name, optimizer_name, gpu, **kwargs)

        self.output_size = vocab_size
        self.max_len = 64
        if 'max_len' in kwargs:
            self.max_len = kwargs['max_len']
        self.sos_id = 1
        if 'sos_id' in kwargs:
            self.sos_id = kwargs['sos_id']
        self.eos_id = 2
        if 'eos_id' in kwargs:
            self.eos_id = kwargs['eos_id']
        '''
        self.use_attention = False
        if 'use_attention' in kwargs:
            self.use_attention = kwargs['use_attention']'''

        self.init_input = None

    def forward_step(self, input_var, hidden, function):
        # input_data: [batch, seq_len]
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)  # [batch, seq_len, embed_dim]
        embedded = self.drop_out(embedded)

        output, hidden = self.model(embedded, hidden)  # [batch_size, seq_len, hidden_dim * num_directions]

        predicted_softmax = function(self.fc_out(output.contiguous()
                                                 .view(-1, self.hidden_dim * self.num_directions)), dim=1).view(
            batch_size,
            output_size,
            -1)
        return predicted_softmax, hidden

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                function=F.log_softmax, teacher_forcing_ratio=0):
        ret_dict = dict()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden,
                                                             function, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output):
            decoder_outputs.append(step_output)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden,
                                                               function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                step_attn = None
                decode(di, step_output)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden,
                                                                   function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output)
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_dim) -> (#layers, #batch, #directions * hidden_dim)
        """
        if self.bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, function, teacher_forcing_ratio):
        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input_data and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length


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
        model = DecoderRNN(vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, num_epochs,
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
