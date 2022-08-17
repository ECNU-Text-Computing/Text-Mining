import datetime
import argparse
import torch
import torch.nn as nn
from base_model import BaseModel


class Transformer(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(Transformer, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                          dropout_rate, learning_rate, num_epochs, batch_size,
                                          criterion_name, optimizer_name, gpu, **kwargs)

        self.num_heads = 2
        if 'num_heads' in kwargs:
            self.num_heads = kwargs['num_heads']
        self.num_encoder_layers = 4
        if 'num_encoder_layers' in kwargs:
            self.num_encoder_layers = kwargs['num_encoder_layers']
        self.num_decoder_layers = 4
        if 'num_decoder_layers' in kwargs:
            self.num_decoder_layers = kwargs['num_decoder_layers']

        self.transformer = nn.Transformer(self.embed_dim, self.num_heads, self.num_encoder_layers,
                                          self.num_decoder_layers, dropout=self.dropout_rate, batch_first=True)
        self.fc_out = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x, y):
        # input: [batch_size, seq_len]
        src = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        tgt = self.embedding(y)  # [batch_size, seq_len, embed_dim]
        hidden = self.transformer(src, tgt)  # [batch_size, seq_len, embed_dim]
        print(hidden.size())
        out = self.fc_out(hidden)  # [batch_size, num_classes]
        return out


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description="Process some description.")
    parser.add_argument('--phase', default='test', help='the function name')
    args = parser.parse_args()

    if args.phase == 'test':
        print("This is a test process.")

        vocab_size, embed_dim, hidden_dim, num_classes, \
        dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu = 100, 64, 32, 2, 0.5, 0.001, 3, 32, 'CrossEntropyLoss', 'Adam', 0

        model = Transformer(vocab_size, embed_dim, hidden_dim, num_classes,
                            dropout_rate, learning_rate, num_epochs, batch_size,
                            criterion_name, optimizer_name, gpu)

        # [batch_size, seq_len] = [3, 5]
        src = torch.LongTensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 4, 2, 7, 5]])
        tgt = torch.LongTensor([[1, 3, 5, 7, 9], [2, 3, 4, 5, 6], [1, 4, 8, 3, 6]])
        out = model(src, tgt)

        print("The output is: {}".format(out))

        print("The test process is done.")

    else:
        print("There is no {} function.".format(args.phase))

    end_time = datetime.datetime.now()
    print("{} takes {} seconds.".format(args.phase, (end_time - start_time).seconds))
    print("Done Transformer.")
