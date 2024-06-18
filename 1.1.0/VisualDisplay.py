
import os
import torch
import torchvision.transforms as transforms
from models.tiny_vit import tiny_vit_21m_224 as create_model
import cv2
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = create_model(num_classes=2, in_chans=3).to(device)
model_weight_path = "./weights/111.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device)["model"])
model.eval()

def cam(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 提取特征图
    features = model.extract_features(image_tensor)
    # 假设我们使用最后一个卷积层的输出作为特征图
    features = features.squeeze(0).detach().cpu().numpy()
    # 生成 CAM
    cam_output = generate_cam(features)

    return cam_output

def generate_cam(features):
    # 假设 features 是一个形状为 (C, H, W) 的数组
    # 生成 CAM 通常涉及到对特征通道的加权求和
    # 这里我们简单地取平均
    cam = np.mean(features, axis=0)
    cam = np.maximum(cam, 0)  # ReLU
    cam = cam / cam.max()     # 归一化
    return cam

def visualize_and_save(image_path, model, save_folder):
    cam_image = cam(image_path, model)
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (224, 224))

    # 将 CAM 映射到热力图
    cam_image = cv2.resize(cam_image, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_image), cv2.COLORMAP_JET)

    final_image = heatmap * 0.4 + original_image * 0.6
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    save_path = os.path.join(save_folder, os.path.basename(image_path))
    cv2.imwrite(save_path, final_image)

def process_folder(folder_path, model, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        visualize_and_save(image_path, model, save_folder)

folder_path = '/home/abelxiaoxing/work/datas/classification/flower_photos/train/daisy'
save_folder = './View'
process_folder(folder_path, model, save_folder)
