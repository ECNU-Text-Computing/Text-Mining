#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
BiLSTM_CRF
======
A class for BiLSTM_CRF.
配置文件：cmed.dl.bilstm_crf.norm.json
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
from sequence_labeling.utils.evaluate_2 import Metrics
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

    def forward(self, x, x_lengths, y):
        batch_size, seq_len = x.size()
        hidden = self._init_hidden(batch_size)
        embeded = self.word_embeddings(x)
        embeded = rnn_utils.pack_padded_sequence(embeded, x_lengths, batch_first=True)
        output, _ = self.lstm(embeded, hidden)  # 使用初始化值
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        out = output.reshape(-1, output.shape[2])  # [batch_size*seq_len, tags_size]
        out = self.output_to_tag(out)

        out = torch.nn.functional.log_softmax(out, dim=1)

        return out

    def test(self, data_path):
        print('Running {} model. Testing...'.format(self.model_name))
        run_mode = 'test'
        best_model_path = '{}{}_{}'.format(self.data_root, self.model_name, self.model_save_path)
        model = torch.load(best_model_path)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            test_data_num = 0
            all_y_predict = []
            all_y_true = []
            for seq_list, tag_list in DataLoader(**self.config).data_generator(data_path=data_path,
                                                                               run_mode=run_mode):
                batch_input, batch_output = self.data_to_index(seq_list, tag_list)
                x, x_len, y, y_len = self.padding(batch_input, batch_output)
                test_data_num += len(x)
                batch_x = torch.tensor(x).long()
                batch_y = torch.tensor(y).long()

                tag_scores = model(batch_x, x_len, y)

                batch_y = batch_y.view(-1)
                loss = self.criterion(tag_scores, batch_y)
                test_loss += loss.item()

                # viterbi解码得到预测。seqs_tag为标签的索引序列。
                score, seqs_tag = self.crf._viterbi_decode(tag_scores)
                y_predict = self.index_to_tag(seqs_tag, y_len)
                all_y_predict = all_y_predict + y_predict
                # print(len(y_predict_list))

                y_true = y.flatten()
                y_true = self.index_to_tag(y_true, y_len)
                all_y_true = all_y_true + y_true
                # print(len(y_true_list))

            # 输出评价结果
            print(Evaluator().classifyreport(all_y_true, all_y_predict))
            f1_score = Evaluator().f1score(all_y_true, all_y_predict)

            # 输出混淆矩阵
            metrix = Metrics(all_y_true, all_y_predict)
            metrix.report_scores()
            metrix.report_confusion_matrix()

            return f1_score


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
