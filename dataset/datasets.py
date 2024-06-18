import torch
import torchvision
class DogCat():
    # 数据增强与图像处理
    IMAGE_SIZE=128
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(1.2*IMAGE_SIZE)),#尺寸调节
        torchvision.transforms.RandomResizedCrop(IMAGE_SIZE),# 随机长宽比裁剪
        torchvision.transforms.RandomHorizontalFlip(),#概率镜像翻转
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
    train_data = torchvision.datasets.ImageFolder("./datas/train/", transform=train_transform)
    test_data=torchvision.datasets.ImageFolder("./datas/test/", transform=valid_transform)

    # 设置batchsize
    batch_size=64

    # 导入数据(一次导入batchsize个)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=8)
