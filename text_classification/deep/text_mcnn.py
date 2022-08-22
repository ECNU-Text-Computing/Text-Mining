import argparse
import datetime
import torch
from base_model import BaseModel
from torch import nn


class TextMCNN(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(TextMCNN, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                       dropout_rate, learning_rate, num_epochs, batch_size,
                                       criterion_name, optimizer_name, gpu, **kwargs)
        self.model_name = 'TextMCNN'
        # 简单CNN参数设置
        # 卷积核数量
        self.num_filters = 3
        if 'num_filters' in kwargs:
            self.num_filters = kwargs['num_filters']
        # 卷积核尺寸
        self.filter_sizes = [3, 4, 5]
        if 'filter_sizes' in kwargs:
            self.filter_sizes = kwargs['filter_sizes']
        self.num_layers = 1
        if 'num_layers' in kwargs:
            self.num_layers = kwargs['num_layers']
        self.convs_layers = nn.ModuleList()

        in_channel = 1
        embed_dim = self.embed_dim
        for i in range(1, self.num_layers - 1):
            convs = nn.ModuleList()
            for filter_size in self.filter_sizes:
                pad_size = filter_size - 2
                if pad_size % 2 == 0:
                    pad_array = (0, 0, int(pad_size / 2), int(pad_size / 2) + 1)
                else:
                    pad_array = (0, 0, int(pad_size / 2) + 1, int(pad_size / 2) + 1)
                each_conv = nn.ModuleList([
                    nn.Sequential(
                        nn.ZeroPad2d(pad_array),
                        nn.Conv2d(in_channels=in_channel,
                                  out_channels=self.num_filters,
                                  kernel_size=(filter_size, embed_dim),
                                  )
                    ),
                    nn.Sequential(
                        nn.LayerNorm(self.num_filters),
                        nn.ReLU(),
                        nn.MaxPool1d(2)
                    )
                ])
                convs.append(each_conv)
            self.num_filters = int(self.num_filters / 2)
            embed_dim = int(self.embed_dim / 2)
            in_channel = 3
            self.convs_layers.append(convs)

        final_convs = nn.ModuleList()
        for filter_size in self.filter_sizes:
            pad_size = filter_size - 2
            if pad_size % 2 == 0:
                pad_array = (0, 0, int(pad_size / 2), int(pad_size / 2) + 1)
            else:
                pad_array = (0, 0, int(pad_size / 2) + 1, int(pad_size / 2) + 1)
            each_conv = nn.ModuleList([
                nn.Sequential(
                    nn.ZeroPad2d(pad_array),
                    nn.Conv2d(in_channels=in_channel,
                              out_channels=self.num_filters,
                              kernel_size=(filter_size, embed_dim),
                              )
                ),
                nn.Sequential(
                    nn.LayerNorm(self.num_filters),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1)
                )
            ])
            final_convs.append(each_conv)
        self.convs_layers.append(final_convs)

    def conv_and_pool(self, x, conv):
        out = conv[0](x).squeeze()  # [batch_size, num_filters, seq_len]
        out = out.permute(0, 2, 1)  # [batch_size, seq_len, num_filters]
        out = conv[1](out).unsqueeze(1)  # [batch_size, 1, num_filters, 1]
        return out

    def forward(self, x):
        # input: [batch_size, seq_len]
        out = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        out = out.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        seq_len = x.size(1)
        batch_size = out.shape[0]
        for convs in self.convs_layers:
            out = torch.cat([self.conv_and_pool(out, conv) for conv in convs], dim=1)
        out = self.drop_out(out.squeeze(-1).reshape(batch_size, -1))  # [batch_size, num_filters*seq_len]
        self.fc = nn.Linear(self.num_filters * seq_len, self.num_classes)
        out = self.fc(out)  # [batch_size, num_classes]
        return out


if __name__ == '__main__':
    if __name__ == '__main__':
        start_time = datetime.datetime.now()
        parser = argparse.ArgumentParser(description='Process some description.')
        parser.add_argument('--phase', default='test', help='the function name.')

        args = parser.parse_args()

        if args.phase == 'test':
            print('This is a test process.')

            # 设置测试用例，验证模型是否能够运行
            # 设置模型参数
            vocab_size, embed_dim, hidden_dim, num_classes, \
            dropout_rate, learning_rate, num_epochs, batch_size, \
            criterion_name, optimizer_name, gpu, num_filters, filter_sizes \
                = 100, 64, 32, 2, 0.5, 0.001, 3, 32, 'CrossEntropyLoss', 'Adam', 0, 3, [4, 5, 6]
            # 创建类的实例
            model = TextMCNN(vocab_size, embed_dim, hidden_dim, num_classes,
                             dropout_rate, learning_rate, num_epochs, batch_size,
                             criterion_name, optimizer_name, gpu, num_filters=num_filters, filter_sizes=filter_sizes)
            # 传入简单数据，查看模型运行结果
            # [batch_size, seq_len] = [3, 5]
            input_data = torch.LongTensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 4, 2, 7, 5]])
            output_data = model(input_data)
            print("The output is: {}".format(output_data))

            print("The test process is done.")
        else:
            print('error! No such method!')
        end_time = datetime.datetime.now()
        print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

        print('Done TextMCNN!')
