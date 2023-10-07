import torch
from torch import nn
from torch.nn import  functional as F

class Resblk(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Resblk,self).__init__()

        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(ch_out)
        self.conv2=nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(ch_out)

        self.extra=nn.Sequential()
        #一个Resize层
        if ch_out!=ch_in:
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self,x):
        fx=self.conv1(x)
        fx=F.relu(self.bn1(fx))
        fx=self.conv2(fx)
        fx=F.relu(self.bn2(fx))

        return fx +self.extra(x)


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet,self).__init__()

        self.conv=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=3,padding=0),
            nn.BatchNorm2d(64)
        )
        self.blk1=Resblk(64,128)
        self.blk2=Resblk(128,256)
        self.blk3=Resblk(256,512)
        self.blk4=Resblk(512,1024)
        #self.outlayer=nn.Linear()

    def forward(self,x):
        x=F.relu(self.conv(x))
        x=self.blk4(self.blk3(self.blk2(self.blk1(x))))

        #打平，给线性层连接
        x=x.view(x.size(0),x.size(1)*x.size(2)*x.size(3))

        self.outlayer=nn.Linear(x.size(1),10)#最后的线性层

        return self.outlayer(x)



def main():
    x=torch.rand(10,3,32,32)
    net=Resnet()
    print(net(x).shape)

if __name__ =='__main__':
    main()