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

class AttentionModule(nn.Module):
    def __init__(self, inputs, n_filters, in_channels, out_channels):
        super(AttentionModule, self).__init__()
        self.inputs = inputs
        self.mean = torch.mean(self.inputs, [1,2], keepdim=True)
        self.conv1 = nn.Conv2d(in_channels, n_filters, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(-1)

    def forward(self, x):
        x = self.mean(x)
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = torch.sigmoid(x)
        x = torch.multiply(self.inputs, x)
        return x 



        
    


