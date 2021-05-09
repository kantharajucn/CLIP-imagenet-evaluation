import torch
import torch.nn as nn


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, pred, targets):
        logsoftmax = nn.LogSoftmax()
        return torch.mean(torch.sum(- targets * logsoftmax(pred), 1))
