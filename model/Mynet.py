from model.resnet import resnet34
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
import torch


class MyNet(nn.Module):
    def __init__(self,resnet_feature):
        super(MyNet,self).__init__()

        self.resnet_feature=resnet_feature
        self.fc = nn.Linear(1024*8,5)
    def forward(self,x):
        x = self.resnet_feature(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return F.log_softmax(x,dim=1)
def Mynetinit():
    # 使用Resnet特征
    resnet = models.resnet34(pretrained=True)
    resnet_modules = list(resnet.children())[:-2]      # 去除最后两层
    resnet_feature = nn.Sequential(*resnet_modules).eval()
    model=MyNet(resnet_feature)
    return model

