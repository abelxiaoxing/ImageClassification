import torch
import torch.nn.functional as F
from model.Mynet import Mynetinit
from dataset.datasets import DogCat
# from model.resnet import resnet50
import os
from tqdm import tqdm
from collections import defaultdict
# from model.resnet import resnet34

device=torch.device("cuda" if torch.cuda.is_available() else "cpu" )#设置训练模式 GPU or CPU


# 加载 ResNet34 模型
model = Mynetinit()

for name, para in model.named_parameters():
    # 除最后的全连接层外，其他权重全部冻结
    if "fc" not in name:
        para.requires_grad_(False)

# 设置新添加的层参数为可训练
for param in model.fc.parameters():
    param.requires_grad = True

# 将模型放在 GPU 上（如果有可用的 GPU）
model.to(device)


pg = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(pg,lr=1e-3,weight_decay=5E-5)




# 设置优化器随epoch数增加而衰减
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,50)
# 定义损失函数
# criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor([2/3, 1/3]))
criterion=torch.nn.CrossEntropyLoss()
criterion=criterion.to(device)

num_classes = 5 # 假设有5个类别
# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    correct = 0
    #加载进度条
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'train_loss_all': 0, 'train_accuracy': 0}

    model.train()
    
    # 统计分类指标的变量
    class_cnt = defaultdict(int)
    tp = defaultdict(int)
    fp = defaultdict(int)
    tn = defaultdict(int)
    fn = defaultdict(int)

    for data, target in train_bar:#加载数据和标签
        data, target = data.to(device), target.to(device) #存入内存或者显存中
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size
        optimizer.zero_grad() #梯度清零避免上一次训练的梯度造成影响
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        output_class = output.argmax(dim=1)

        #更新分类指标统计结果
        for cls in range(num_classes):
            class_cnt[cls] += (target == cls).sum().item()
            tp[cls] += ((output_class == cls) & (target == cls)).sum().item()
            fp[cls] += ((output_class == cls) & (target != cls)).sum().item()
            tn[cls] += ((output_class != cls) & (target != cls)).sum().item()
            fn[cls] += ((output_class != cls) & (target == cls)).sum().item()

        #计算loss
        running_results['train_loss_all'] += loss.item() * batch_size
        #计算正确数
        correct += output_class.eq(target.view_as(output_class)).sum().item()
        #计算正确率
        running_results['train_accuracy'] = 100 * correct / running_results['batch_sizes']
        
        ##更新显示训练情况
        train_bar.set_description(desc='[%d/%d] train_accuracy: %.4f train_loss: %.4f' % (epoch, num_epochs, running_results['train_accuracy'], running_results['train_loss_all']/running_results['batch_sizes']))
    
    # 打印每一类的分类指标
    for cls in range(num_classes):
        # precision = tp[cls] / (tp[cls] + fp[cls])
        if tp[cls] + fp[cls] > 0:
            precision = tp[cls] / (tp[cls] + fp[cls])
        else:
            precision = 0
            print(tp[cls],fp[cls])
        # recall = tp[cls] / (tp[cls] + fn[cls])
        if tp[cls] + fn[cls] > 0:
            recall = tp[cls] / (tp[cls] + fn[cls])
        else:
            precision = 0
            print(tp[cls],fn[cls])
        print('Class %d precision: %.4f recall: %.4f' % (cls, precision, recall))

    torch.save(model.state_dict(), './pth/'+'%d'%epoch+'.pth') #每个epoch都保存模型
    # 首次训练10个epoch后解冻网络参数
    if epoch == 50:
        for name, para in model.named_parameters():
            para.requires_grad_(True)

def val(model, device, test_loader):
    val_bar = tqdm(test_loader)
    valing_results = {'val_loss': 0, 'val_accuracy': 0}

    # 统计分类指标的变量
    class_cnt = defaultdict(int)
    tp = defaultdict(int)
    fp = defaultdict(int)
    tn = defaultdict(int)
    fn = defaultdict(int)

    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in val_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # 更新分类指标统计结果
            output_class = output.argmax(dim=1)
            for cls in range(num_classes):
                class_cnt[cls] += (target == cls).sum().item()
                tp[cls] += ((output_class == cls) & (target == cls)).sum().item()
                fp[cls] += ((output_class == cls) & (target != cls)).sum().item()
                tn[cls] += ((output_class != cls) & (target != cls)).sum().item()
                fn[cls] += ((output_class != cls) & (target == cls)).sum().item()

            valing_results['val_loss'] += criterion(output, target).item()
            output_class = output.argmax(dim=1)           
            correct += output_class.eq(target.view_as(output_class)).sum().item()
            valing_results['val_accuracy'] = 100 * correct / len(DogCat.test_data)
            val_bar.set_description(desc='[%d/%d] val_accuracy: %.4f val_loss: %.4f' % (epoch, num_epochs, valing_results['val_accuracy'], valing_results['val_loss']/len(DogCat.test_data)))
    
    # 打印每一类的分类指标
    for cls in range(num_classes):
        if tp[cls] + fp[cls] > 0:
            precision = tp[cls] / (tp[cls] + fp[cls])
        else:
            precision = 0
            print(tp[cls],fp[cls])
        if tp[cls] + fn[cls] > 0:
            recall = tp[cls] / (tp[cls] + fn[cls])
        else:
            precision = 0
            print(tp[cls],fn[cls])
        print('Class %d precision: %.4f recall: %.4f' % (cls, precision, recall))
        
        
num_epochs = 1000
for epoch in range(num_epochs):
    train(model, device, DogCat.train_loader, optimizer, epoch)
    val(model, device, DogCat.test_loader)
    scheduler.step() #优化器学习率衰减
