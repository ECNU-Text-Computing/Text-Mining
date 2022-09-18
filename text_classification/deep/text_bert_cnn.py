import argparse
import datetime
import torch
import torch.nn as nn
from deep.base_model import BaseModel
from transformers import BertModel


# BERT2+CNN模型
class BERTCNN(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(BERTCNN, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                      dropout_rate, learning_rate, num_epochs, batch_size,
                                      criterion_name, optimizer_name, gpu, **kwargs)
        # 基本参数设置
        # bert模型的输出向量为768维
        self.hidden_dim = 768
        self.embed_dim = 768
        # 加载的预训练模型名称
        self.model_path = 'bert-base-uncased'
        if 'model_path' in kwargs:
            self.model_path = kwargs['model_path']
        # 隐藏层维度设置
        self.hidden_dim2 = self.hidden_dim // 2
        if 'hidden_dim2' in kwargs:
            self.hidden_dim2 = kwargs['hidden_dim2']

        # bert模型设置
        self.bert = BertModel.from_pretrained(self.model_path, return_dict=True, output_attentions=True,
                                              output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad = True

        # 卷积核的数量
        self.out_channels = 8
        if 'out_channels' in kwargs:
            self.out_channels = kwargs['out_channels']
        # 卷积核的大小。数据类型为列表，列表长度即为卷积核数量
        self.filter_sizes = [2, 3, 4]
        if 'filter_sizes' in kwargs:
            self.filter_sizes = kwargs['filter_sizes']
        # 设置CNN模型的参数。由于可能存在多个卷积核，因此需构建多个模型，此处采用moduel list存储不同的模型
        self.cnns = nn.ModuleList()
        for filter_size in self.filter_sizes:
            self.cnns.append(nn.Sequential(
                nn.Conv2d(1, self.out_channels, (filter_size, 768)),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d(1)
            ))
        # 设置全连接层的参数
        self.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.out_channels * len(self.filter_sizes), self.hidden_dim2),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_dim2, self.num_classes)
        )

    def forward(self, x):
        # x: [batch_size, seq_len]
        output = self.bert(x)  # [batch_size, hidden_dim]
        # [batch_size, 1, seq_len, hidden_dim]
        cnn_input = output['last_hidden_state'].unsqueeze(1)
        # [batch_size, out_channels * len(filter_sizes)]
        cnn_output = torch.cat([conv(cnn_input).squeeze(-1).squeeze(-1) for conv in self.cnns], dim=-1)
        out = self.fc(cnn_output)  # [batch_size, num_classes]
        return out


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
        vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu, out_channels, filter_sizes \
            = 100, 64, 32, 2, 0.5, 0.0001, 3, 3, 'CrossEntropyLoss', 'Adam', 0, 2, [2, 3, 4]

        # 测试所用为2通道3卷积核的CNN
        model = BERTCNN(vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, num_epochs,
                        batch_size, criterion_name, optimizer_name, gpu,
                        out_channels=out_channels, filter_sizes=filter_sizes)

        input_data = torch.LongTensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
                                       [1, 3, 5, 7, 9], [2, 4, 6, 8, 10],
                                       [1, 4, 8, 3, 6]])  # [batch_size, seq_len] = [6, 5]

        output_data = model(input_data)
        print(output_data)
        print('The test process is done!')

    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done BERTCNN Model!')
