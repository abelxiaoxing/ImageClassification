import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from models.ConvNext_V2 import convnextv2_atto as create_model

num_classes = 5
img_size = 224
with open("./class_indices.json", "r") as f:
    class_indict = json.load(f)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_model(num_classes=num_classes).to(device)
model_weight_path = "./pre_weights/test.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device)["model"])
model.eval()

def read_imagefile(file):
    image = Image.open(file)
    return image

def transform_image(image_bytes: Image.Image,img_size):
    data_transform = transforms.Compose(
        [
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return data_transform(image_bytes).unsqueeze(0)

def get_prediction(image_bytes: Image.Image,img_size):
    img = transform_image(image_bytes=image_bytes,img_size=img_size).to(device)
    with torch.no_grad():
        outputs = torch.softmax(torch.squeeze(model(img)),dim=0)
    return outputs



def main():

    print(f"using {device} device.")
    img = read_imagefile("./sunflowers.jpg")
    plt.imshow(img)
    outputs=get_prediction(image_bytes=img,img_size=img_size)
    predict_pro, predict_cla = outputs.max(0)
    print_res = "class: {}   prob: {:.3f}".format(class_indict[str(predict_cla.item())], predict_pro.item())
    plt.title(print_res)
    for i in range(len(outputs)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], outputs[i]))
    plt.show()


if __name__ == "__main__":
    main()
