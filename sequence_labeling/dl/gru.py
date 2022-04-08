import argparse
import datetime
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.data_processor import DataProcessor
from sequence_labeling.dl.base_model import BaseModel

torch.manual_seed(1)


class GRU(BaseModel):

    def __init__(self, **config):
        super(GRU, self).__init__(**config)
        self.config = config
        self.data_root = config['data_root']
        self.model_save_path = config['model_save_name']

        self.pad_token = config['pad_token']

        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.bidirectional = True if config['bidirectional'] == 'True' else False
        self.num_layers = config['num_layers=']
        self.tagset_size = config['tag_size']
        self.num_epochs = config['num_epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout_rate']

        self.criterion_dict = {
            'NLLLoss': torch.nn.NLLLoss,
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss
        }
        self.criterion_name = config['criterion_name']
        if self.criterion_name not in self.criterion_dict:
            raise ValueError("There is no criterion_name: {}.".format(self.criterion_name))

        self.optimizer_dict = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam
        }
        self.optimizer_name = config['optimizer_name']
        if self.optimizer_name not in self.optimizer_dict:
            raise ValueError("There is no optimizer_name: {}.".format(self.optimizer_name))

        vocab = DataProcessor(**self.config).load_vocab()
        vocab_size = len(vocab)
        padding_idx = vocab[self.pad_token]
        self.word_embeddings = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=padding_idx)

        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional,
                          dropout=self.dropout)  # GRU循环神经网络

        # The linear layer that maps from hidden state space to tag space
        self.n_directions = 2 if self.bidirectional else 1  # 双向循环，输出的hidden是正向和反向hidden的拼接，所以要 *2
        self.hidden_to_tag = nn.Linear(self.hidden_dim * self.n_directions, self.tagset_size)  # 全连接层

    def _init_hidden(self, batch_size):  # 初始化h_0
        return torch.zeros(self.num_layers * self.n_directions, batch_size, self.hidden_dim)

    def forward(self, X, X_lengths, run_mode):
        batch_size, seq_len = X.size()
        hidden = self._init_hidden(batch_size)
        X = self.word_embeddings(X)
        X = rnn_utils.pack_padded_sequence(X, X_lengths, batch_first=True)
        X, _ = self.gru(X, hidden)
        X, _ = rnn_utils.pad_packed_sequence(X, batch_first=True)
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        X = self.hidden_to_tag(X)

        if run_mode == 'train':
            tag_scores = F.log_softmax(X, dim=1)  # tag_scores的shape为[batch_size*seq_len, tagset_size]
            # print('shape of tag_scores:{}'.format(tag_scores.shape))
            return tag_scores
        elif run_mode == 'eval' or 'test':
            tag_scores = F.log_softmax(X, dim=1)
            # print('标注结果转换为Tag索引序列：', torch.max(scores, dim=1))
            predict = list(torch.max(tag_scores, dim=1)[1].numpy())  # [batch_size, seq_len]大小的列表
            return predict
        else:
            raise RuntimeError("main.py调用model.run_model()时，参数'run_mode'未赋值！")


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Base_Model!')
