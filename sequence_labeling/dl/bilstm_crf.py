#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
BiLSTM_CRF
======
A class for BiLSTM_CRF.
"""

import argparse
import datetime
import sys

import torch
import torch.nn.utils.rnn as rnn_utils

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from sequence_labeling.dl.base_model import BaseModel
from sequence_labeling.utils.evaluate import Evaluator
from sequence_labeling.data_loader import DataLoader
from .crf import CRF

torch.manual_seed(1)


class BiLSTM_CRF(BaseModel):

    def __init__(self, **config):
        super().__init__(**config)

        self.crf = CRF(self.tags_size, self.tag_index_dict)
        self.criterion = self.crf.neg_log_likelihood

    def _init_hidden(self, batch_size):  # 初始化h_0 c_0
        hidden = (torch.zeros(self.layers * self.n_directions, batch_size, self.hidden_dim),
                  torch.zeros(self.layers * self.n_directions, batch_size, self.hidden_dim))
        return hidden

    def forward(self, X, X_lengths):
        batch_size, seq_len = X.size()
        hidden = self._init_hidden(batch_size)
        embeded = self.word_embeddings(X)
        embeded = rnn_utils.pack_padded_sequence(embeded, X_lengths, batch_first=True)
        output, _ = self.lstm(embeded, hidden)  # 使用初始化值
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        out = output.contiguous()
        out = out.view(-1, out.shape[2])
        out = self.output_to_tag(out)

        return out

    def test(self):
        print('Running {} model. Testing...'.format(self.model_name))
        run_mode = 'test'
        best_model_path = '{}{}_{}'.format(self.data_root, self.model_name, self.model_save_path)
        model = torch.load(best_model_path)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            test_data_num = 0
            for x, x_len, y, y_len in DataLoader(**self.config).data_generator(data_path=self.data_root,
                                                                               run_mode=run_mode):
                test_data_num += len(x)
                batch_x = torch.tensor(x).long()
                tag_scores = model(batch_x, x_len)

                output = torch.nn.functional.log_softmax(tag_scores, dim=1)  # [batch_size*seq_len, tags_size]
                batch_y = torch.tensor(y).long()
                batch_y = batch_y.view(-1)
                loss = self.criterion(output, batch_y)
                test_loss += loss.item()

                # viterbi解码得到预测。seqs_tag为标签的索引序列。
                score, seqs_tag = self.crf._viterbi_decode(tag_scores)
                y_predict = self.index_to_tag(seqs_tag)
                print(y_predict)

                y_true = y.flatten()
                y_true = self.index_to_tag(y_true)
                print(y_true)

                # 输出评价结果
                print(Evaluator().classifyreport(y_true, y_predict))


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
