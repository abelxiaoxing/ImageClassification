import torch
from models.tiny_vit import tiny_vit_21m_224 as create_model
import cv2
# from models.ConvNext_V2 import convnextv2_tiny as create_model
import os
import shutil
from PIL import Image
from torchvision import transforms
import numpy as np

class WhiteBalance:
    def __call__(self,img):
        wb=cv2.xphoto.createSimpleWB()
        img=np.array(img)
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img=wb.balanceWhite(img)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=Image.fromarray(img)
        return img

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    img_size = 224
    data_transform = transforms.Compose(
        [
            WhiteBalance(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_path="/home/abelxiaoxing/work/datas/package/package_test/val/EmptyBag"
    empty_path = os.path.join(os.path.dirname(img_path), "Empty")
    non_empty_path = os.path.join(os.path.dirname(img_path), "Non-Empty")
    os.makedirs(empty_path, exist_ok=True)
    os.makedirs(non_empty_path, exist_ok=True)

    # model = torch.jit.load("./weights/best_model.pth").to(device)
    model = create_model(num_classes=num_classes,in_chans=3).to(device)
    model_weight_path = "./pre_weights/checkpoint-21.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device)["model"])
    model.eval()

    for file_name in os.listdir(img_path):
        file_path = os.path.join(img_path, file_name)
        img = Image.open(file_path).convert("RGB")
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)

        predicted_class_index = torch.argmax(predict).item()

        if predicted_class_index == 0:
            empty_file_path = os.path.join(empty_path, file_name)
            shutil.move(file_path, empty_file_path)
        elif predicted_class_index == 1:
            non_empty_file_path = os.path.join(non_empty_path, file_name)
            shutil.move(file_path, non_empty_file_path)

        # print("Image: {:10}   class: {:10}   prob: {:.3}".format(file_name, predicted_class, predict[predicted_class_index].numpy()))


if __name__ == "__main__":
    main()
