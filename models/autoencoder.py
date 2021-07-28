import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, input_channel_num):
        super(AutoEncoder, self).__init__()
        self.conv_size = 3
        self.pad_size = int((self.conv_size - 1) / 2)
        self.feature_num = input_channel_num * 2
        self.conv_1 = nn.Conv2d(input_channel_num, self.feature_num, self.conv_size, 1, padding=self.pad_size)
        self.conv_1.weight = torch.nn.Parameter(torch.zeros_like(self.conv_1.weight))
        self.conv_1.bias = torch.nn.Parameter(torch.zeros_like(self.conv_1.bias))

        self.conv_2 = nn.Conv2d(self.feature_num, 1, self.conv_size, 1, padding=self.pad_size)
        self.conv_2.weight = torch.nn.Parameter(torch.zeros_like(self.conv_2.weight) / 100)
        self.conv_2.bias = torch.nn.Parameter(torch.zeros_like(self.conv_2.bias))

        self.conv_3 = nn.Conv2d(1, input_channel_num, 1, 1, padding=0)
        self.conv_3.weight = torch.nn.Parameter(torch.zeros_like(self.conv_3.weight))
        self.conv_3.bias = torch.nn.Parameter(torch.zeros_like(self.conv_3.bias))

        self.en_conv = nn.Sequential(
            self.conv_1,
            nn.Sigmoid(),
            self.conv_2,
            nn.Sigmoid(),
        )
        self.de_conv = nn.Sequential(
            self.conv_3,
            nn.Sigmoid()
        )

    def forward(self, x):
        code = self.en_conv(x)
        decoded = self.de_conv(code)
        return code, decoded

    def __str__(self):
        return 'ae_{}'.format(self.conv_size)