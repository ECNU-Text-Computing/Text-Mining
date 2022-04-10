
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

from Deep.Base_Model import Base_Model

class FastText(Base_Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(FastText, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)

        # 分别随机初始化 bi-gram tri-gram对应的词嵌入矩阵
        self.embedding_ngram2 = nn.Embedding(vocab_size, embed_dim)
        self.embedding_ngram3 = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        #隐层
        self.fc1 = nn.Linear(embed_dim * 3, hidden_dim)
        # self.dropout2 = nn.Dropout(config.dropout)
        #输出层
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x(uni-gram,seq_len,bi-gram,tri-gram)
        #  基于 uni-gram,seq_len,bi-gram,tri-gram对应的索引，在各个词嵌入矩阵中查询，得到词嵌入 [batch_size,seq_len,embed_dim]
        print(x.size())
        #进行bow
        out_word = self.embedding(x)
        print(out_word.size())
        out_bigram = self.embedding_ngram2(x)
        print(out_bigram.size())
        out_trigram = self.embedding_ngram3(x)
        print(out_trigram.size())
        #三种嵌入进行拼接
        out = torch.cat((out_word, out_bigram, out_trigram), -1)   # [batch_size,seq_len,embed_dim*3]
        print(out.size())
        #沿长度维 作评价
        out = out.mean(dim=1)   # [batch_size，embed*3]
        print(out.size())
        out = self.dropout(out)
        out = self.fc1(out)  #[batch_size,hidden_dim]
        print(out.size())
        out = F.relu(out)
        out = self.fc2(out)  #[batch_size,num_classes]
        print(out.size())
        return out


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        # For testing our model, we can set the hyper-parameters as follows.
        vocab_size, embed_dim, hidden_dim, num_classes,\
        dropout_rate, learning_rate, num_epochs, batch_size,\
        criterion_name, optimizer_name, gpu= 200, 64, 64, 2,0.5, 0.0001, 2, 64, 'CrossEntropyLoss', 'Adam',0

        # new an objective.
        model = FastText(vocab_size, embed_dim, hidden_dim, num_classes,
                         dropout_rate, learning_rate, num_epochs, batch_size,
                         criterion_name, optimizer_name,gpu)

        # a simple example of the input.
        input = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],[4,5,6,7,0]]
        input = torch.LongTensor(input)  # input: [batch_size, seq_len] = [4, 5]

        # the designed model can produce an output.
        out = model(input)
        print(out)
        print('This is a test process.')
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Base_Model!')
