import torch
import torch.nn.functional as F
import torchvision
# 数据增强与图像处理
IMAGE_SIZE=256
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(int(1.2*IMAGE_SIZE)),#尺寸调节
    torchvision.transforms.RandomHorizontalFlip(),#概率镜像翻转
    torchvision.transforms.RandomResizedCrop(IMAGE_SIZE),# 随机长宽比裁剪
    torchvision.transforms.ToTensor(),# 把ndarray图片格式转化为tensor并归一化
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])# 标准化:先减均值再除以标准差
])
# 对验证集的图片进行处理变换
valid_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),# 随机长宽比裁剪
    torchvision.transforms.ToTensor(),# 把ndarray图片格式转化为tensor并归一化
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])# 标准化:先减均值再除以标准差
])
# 读取数据
train_data = torchvision.datasets.ImageFolder("./flower_photos/train/", transform=train_transform)
test_data=torchvision.datasets.ImageFolder("./flower_photos/test/", transform=valid_transform)

# 设置batchsize
batch_size=128

# 导入数据(一次导入batchsize个)
device=torch.device("cuda")
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=8)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=8)
# 定义网络模型
class CNN(torch.nn.Module):# 新建一个网络类，就是需要搭建的网络，必须继承PyTorch的nn.Module父类
    def __init__(self):# 构造函数，用于设定网络层
        super(CNN,self).__init__()# 标准语句
        self.conv1 = torch.nn.Sequential(#设置一个网络小合集,conv1将会按照一下五个网络依次执行
            torch.nn.Conv2d(3,16,kernel_size=3,padding=1),# 第一个卷积层，输入通道数3，输出通道数16，卷积核大小3×3，padding大小1，其他参数默认
            torch.nn.ReLU(),# 第一次卷积结果经过ReLU激活函数处理,将线性变化转化为非线性变化
            torch.nn.BatchNorm2d(16),# 标准化,加速网络收敛速度
            torch.nn.MaxPool2d(2),# 第一次池化，池化大小2×2，方式Max pooling,shape为128×128×16
            torch.nn.Dropout(0.25)# 抛弃25%部分的神经元,避免过拟合
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,kernel_size=3,padding=1),# 第二个卷积层，输入通道数16，输出通道数32，卷积核大小3×3，padding大小0，其他参数默认
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2),# 池化kernel大小为2,步长默认为kernel的大小,padding默认为0，方式Max pooling,shape为64×64×32
            torch.nn.Dropout(0.25)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,kernel_size=3,padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2),# 池化大小2×2，方式Max pooling,shape为32×32×64
            torch.nn.Dropout(0.25)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,kernel_size=3,padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2),# 池化大小2×2，方式Max pooling,shape为16×16×64
            torch.nn.Dropout(0.25)
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,kernel_size=3,padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2),# 池化大小2×2，方式Max pooling,shape为8×8×64
            torch.nn.Dropout(0.25)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64*8*8,256),# 第一个全连层，线性连接，输入节点数8×8×64，输出节点数256
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256,5),# 第三个全连层，线性连接，输入节点数256，输出节点数5
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0),-1)# 由于全连层输入的是一维张量，因此需要对输入的[64×64×16]格式数据排列成向量形式
        x = self.fc(x)
        return F.log_softmax(x,dim=1)
#设置训练方式CPU or GPU
device=torch.device("cuda" if torch.cuda.is_available() else "cpu" )
model=CNN()
# model.load_state_dict(torch.load('./pth/110.pth'))#加载预训练模型
model=model.to(device)


# 设置参数优化器
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
#optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
# 设置优化器随epoch的调整(看具体数据集,个人喜好余弦衰退)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,50)
# 定义损失函数
criterion=torch.nn.CrossEntropyLoss()
criterion=criterion.to(device)

def train(model,device,train_loader,optimizer,epoch):
    model.train()
    correct=0
    for data,target in train_loader:#加载数据和标签
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad() #梯度清零避免上一次训练的梯度造成影响
        output=model(data)#输出预测值
        loss=criterion(output,target)#loss计算
        loss.backward()#反向传播
        optimizer.step()
        output_class=output.argmax(dim=1)#获得类别信息
        correct+=output_class.eq(target.view_as(output_class)).sum().item()#计算预测正确的数量
    acc=100*correct/len(train_data)#计算训练集准确率
    torch.save(model.state_dict(),'./pth/'+'%d'%epoch+'.pth')#每个epoch都保存模型
    print("epoch:{},train_loss:{},train_accuracy:{}".format(epoch,loss.mean(),acc))

def val(model,device,test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in (test_loader):
            data,target=data.to(device),target.to(device)
            output=model(data)
            test_loss+=criterion(output,target).item()
            output_class=output.argmax(dim=1)           
            correct+=output_class.eq(target.view_as(output_class)).sum().item()
    acc=100*correct/len(test_data)#计算测试集准确率
    print("test_loss:{},test_accuracy:{}".format(test_loss,acc))

num_epochs=200
for epoch in range(num_epochs):
    train(model,device,train_loader,optimizer,epoch)
    val(model,device,test_loader)
    scheduler.step()#优化器学习率衰减