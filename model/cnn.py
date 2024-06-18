import torch
import torch.nn.functional as F

# 定义网络模型
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,kernel_size=3,padding=1,stride=1),#输入channel为3,输出为16,卷积核大小为3,padding为1,步长为1
            torch.nn.Conv2d(16,16,kernel_size=3,padding=1,stride=2),#输入channel为16,输出为16,卷积核大小为3,padding为0,步长为2,压缩特征长宽
            torch.nn.BatchNorm2d(16),#归一化处理,括号内为channel数
            torch.nn.ReLU(),#Relu非线性化
            torch.nn.MaxPool2d(2)#以2*2进行最大池化,压缩特征长宽
        )
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,kernel_size=3,padding=1,stride=1),
            torch.nn.Conv2d(32,32,kernel_size=3,padding=1,stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )        

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,kernel_size=3,padding=1,stride=1),
            torch.nn.Conv2d(64,64,kernel_size=3,padding=1,stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.fc1 = torch.nn.Linear(2*2*64,10)#全连接层,将2*2*64个神经元连接输出到10个神经元上
        self.dropout = torch.nn.Dropout(0.1)#将一半参数抛弃,避免过拟合.
        self.fc2 = torch.nn.Linear(10,2)

    def forward(self,x):
        out =self.layer1(x)
        out =self.layer2(out)
        out =self.layer3(out)
        out =out.view(out.size(0),-1)#将特征铺平为向量形式,便于进行全连接
        out =F.relu(self.fc1(self.dropout(out)))
        out =self.fc2(out)
        return F.log_softmax(out,dim=1)# log_softmax进行非线性化,这个激活函数用于分类任务效果较好
