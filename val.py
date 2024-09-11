
import os
import shutil
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from timm.models import create_model
import cv2
import numpy as np


# 每行最低亮度为0,全局最高亮度为255进行均匀拉伸
class ObjectEnhancement:
    def __call__(self, img):
        img_np = np.array(img)
        balanced_img = np.zeros_like(img_np, dtype=np.uint8)
        for channel in range(img_np.shape[2]):
            channel_data = img_np[:, :, channel]
            max_val = np.max(channel_data)
            min_val_per_row = np.min(channel_data, axis=1)
            for i in range(img_np.shape[0]):
                row = channel_data[i, :]
                if max_val - min_val_per_row[i] != 0:
                    balanced_img[i, :, channel] = 255 / (max_val - min_val_per_row[i]) * (row - min_val_per_row[i])
                else:
                    balanced_img[i, :, channel] = 0
        balanced_img = Image.fromarray(balanced_img)
        return balanced_img

class WhiteBalance:
    def __call__(self, img):
        wb = cv2.xphoto.createSimpleWB()
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = wb.balanceWhite(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img

# 初始化模型的辅助函数
def initialize_model(model_weight_path, device):
    model = torch.load(model_weight_path, map_location=device)["model"]
    model.eval()
    return model

# 创建数据变换的辅助函数
def create_data_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # WhiteBalance(),
        ObjectEnhancement(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# 根据模型预测移动图像
def val_move(img_path, model_weight_path, img_size, device):

    empty_path = os.path.join(os.path.dirname(img_path), "Empty")
    non_empty_path = os.path.join(os.path.dirname(img_path), "NonEmpty")
    os.makedirs(empty_path, exist_ok=True)
    os.makedirs(non_empty_path, exist_ok=True)

    data_transform = create_data_transform(img_size)
    model = initialize_model(model_weight_path, device)

    for file_name in os.listdir(img_path):
        file_path = os.path.join(img_path, file_name)
        img = Image.open(file_path).convert("RGB")
        img = data_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = torch.squeeze(model(img)).cpu()
            predict = torch.softmax(output, dim=0)

        predicted_class_index = torch.argmax(predict).item()
        target_path = empty_path if predicted_class_index == 0 else non_empty_path
        shutil.move(file_path, os.path.join(target_path, file_name))

# 计算并打印精确率和召回率
def val_precision(img_path, model_name, model_weight_path, img_size, device):
    data_transform = create_data_transform(img_size)
    dataset = ImageFolder(root=img_path, transform=data_transform)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)
    criterion = torch.nn.CrossEntropyLoss()

    model = initialize_model(model_name, 2, model_weight_path, device)

    true_positives = [0] * 2
    false_positives = [0] * 2
    false_negatives = [0] * 2

    for images, target in data_loader:
        images, target = images.to(device), target.to(device)
        output = model(images)
        loss = criterion(output, target)
        _, preds = torch.max(output, 1)

        for i in range(2):
            true_positives[i] += torch.sum((preds == i) & (target == i)).item()
            false_positives[i] += torch.sum((preds == i) & (target != i)).item()
            false_negatives[i] += torch.sum((preds != i) & (target == i)).item()

    for i in range(2):
        precision = true_positives[i] / (true_positives[i] + false_positives[i]) if true_positives[i] + false_positives[i] > 0 else 0
        recall = true_positives[i] / (true_positives[i] + false_negatives[i]) if true_positives[i] + false_negatives[i] > 0 else 0
        print(f'Precision{i}: {precision:.4f}, Recall{i}: {recall:.4f}')

if __name__ == "__main__":

    img_path = "/home/abelxiaoxing/work/datas/package/package/val/EmptyBag"
    model_weight_path = "checkpoint-68.pth"
    img_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Start calculation!")
    val_move(img_path, model_weight_path, img_size, device)
    # val_precision(img_path, model_weight_path, img_size, device)
