import torch


class BasicModule(torch.nn.Module):
    """ 加载保存模型 """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = './checkpoints/' + self.model_name
        torch.save(self.state_dict(), prefix)
        return name
