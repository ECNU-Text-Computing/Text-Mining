
import warnings
import torch


class DefaultConfig(object):

    model = 'GAT'
    use_gpu = False
    if use_gpu:
        device = 'gpu'
    else:
        device = 'cpu'
    load_model_path = None
    # 以上模型基本信息

    network = 'cora'        # 网络名称
    batch_size = 8          # 批次大小
    nhid = 8                # 隐藏层维度
    nheads = 8              # 注意力头数, 针对GAT
    alpha = 0.2             # LeakyReLU的负斜率, 针对GAT
    num_layers = 2          # 网络层数, 针对GraphSage
    num_workers = 3         # 工作线程数
    max_epoch = 50          # 训练轮次
    lr = 0.01               # 学习率
    lr_decay = 0.9          # 学习率衰减率
    weight_decay = 1e-5     # 设置权重衰减系数
    train_rate = 0.05
    val_rate = 0.05
    dropout = 0.6           # dropout概率

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()