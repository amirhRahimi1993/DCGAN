from statistics import mode
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self):
        pass
    def block(self,in_channel,out_channel,Kernel_size,stride,padding,BIAS=False,down=True):
        if down:
            Sequence = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,Kernel_size,stride,padding,bias=BIAS),
                nn.LeakyReLU(0.2),
                )
        else:
            Sequence= nn.Sequential(
            nn.ConvTranspose2d(
                in_channel,
                out_channel,
                Kernel_size,
                stride,
                padding,
                bias=BIAS,
            ),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            )
        return Sequence
class Discriminator(nn.Module):
    def __init__(self,input_channel,feature):
        super(Discriminator,self).__init__()
        DOWN= True
        down_block = Block()
        self.disc= nn.Sequential(
            down_block.block(input_channel,feature,Kernel_size=4,stride=2,padding=1,BIAS=False,down=DOWN),
            down_block.block(feature,feature*2,Kernel_size=4,stride=2,padding=1,BIAS=True,down=DOWN),
            down_block.block(feature*2,feature*4,Kernel_size=4,stride=2,padding=1,BIAS=True,down=DOWN),
            down_block.block(feature*4,feature*8,Kernel_size=4,stride=2,padding=1,BIAS=True,down=DOWN),
            nn.Conv2d(feature*8,1,kernel_size=4,stride=2,padding=0),
            nn.Sigmoid(),
            )
    def forward(self,x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self,input_channel,channel_img,feature_g):
        super(Generator,self).__init__()
        up_block= Block()
        DOWN = False
        self.net= nn.Sequential(
            up_block.block(input_channel,feature_g*16,Kernel_size=4,stride=1,padding=0,BIAS=False,down=DOWN),
            up_block.block(feature_g*16,feature_g*8,Kernel_size=4,stride=2,padding=1,BIAS=False,down=DOWN),
            up_block.block(feature_g*8,feature_g*4,Kernel_size=4,stride=2,padding=1,BIAS=False,down=DOWN),
            up_block.block(feature_g*4,feature_g*2,Kernel_size=4,stride=2,padding=1,BIAS=False,down=DOWN),
            nn.ConvTranspose2d(
                feature_g * 2 , channel_img,kernel_size=4,stride=2,padding=1
                ),
                nn.Tanh(),
            )
    def forward(self,x):
        return self.net(x)
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal(m.weight.data,0.0,0.2)

def test():
    N, in_channels, H, W = 8,3,64,64
    noise_dim= 100
    x= torch.randn((N,in_channels,H,W))
    disc= Discriminator(in_channels,8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
#test()