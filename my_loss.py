import torch
import torch.nn as nn


class MyLoss(nn.Module):
    def __init__(self, MyLoss_weight=0.1):
        super(MyLoss, self).__init__()
        self.MyLoss_weight = MyLoss_weight

    def forward(self, x, y):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.abs((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]) - (y[:, :, 1:, :] - y[:, :, :h_x - 1, :])).sum()
        w_tv = torch.abs((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]) - (y[:, :, :, 1:] - y[:, :, :, :w_x - 1])).sum()
        return self.MyLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

