import torch 
import torch.nn as nn 
import torchvision.transforms.functional as TF 

class DilatedConv(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(DilatedConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size= 9, stride= 1, padding= 0, dilation= 2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


    