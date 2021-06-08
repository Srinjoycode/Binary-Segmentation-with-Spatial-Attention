import torch 
import torch.nn as nn 
import torchvision.transforms.functional as TF 

class UpDilatedConv(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(UpDilatedConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size= 3, stride= 2, padding= 0, dilation= 2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)



class DownDilatedConv(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(DownDilatedConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size= 4, stride= 4, padding= 0, dilation= 2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)
