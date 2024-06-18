import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
#from model.SwinTransformer import swin_tiny_patch4_window7_224 as create_model
#from model.resnet import resnet152 as create_model
from model.tiny_vit import tiny_vit_21m_224 as create_model
import os
from tqdm import tqdm
from collections import defaultdict
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2 
IMAGE_SIZE=224

def inference(model, device, img_path, picture_id):
    img = Image.open(img_path).convert('RGB')
    transform =transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)
    img = img.to(device)
    model.eval()
    with torch.no_grad():
        output = model(img)
        output_class = output.argmax(dim=1)
    class_str = str(output_class.item())
    print('Prediction: Class %s',class_str)
    return class_str

def model_init(model_path="./pth/best_model.pth"):
    model = create_model(num_classes=num_classes)   
    model.load_state_dict(torch.load(model_path, map_location=device)['model'])
    model = model.to(device)
    return model

def detection(img_path, picture_id,model):
    output_class=inference(model, device, img_path, picture_id)
    return output_class,picture_id

# model = model_init(model_path="./pth/best_model.pth")
# detection(img_path="D:\\PackageClassification1.0\\datas\\val\\Non-emptyBag\\0517185537014_botton.bmp", picture_id="01111")