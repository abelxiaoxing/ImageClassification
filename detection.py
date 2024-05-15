import torchvision.transforms as transforms
import torch
from PIL import Image
import time


@torch.no_grad()
def inference(model, img):
    model.eval()
    stime = time.time()
    output = model(img)
    etime = time.time()
    print("time is ", etime - stime)
    output_class = output.argmax(dim=1)
    class_str = str(output_class.item())
    return class_str


def model_init(model_path=r"weights/best_model.pth"):
    model = torch.jit.load(model_path)
    model = model.to(device)
    return model


def detection(img_path, picture_id, model):
    IMAGE_SIZE = 224
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    output_class = inference(model, img)
    print(output_class)
    return output_class, picture_id


def net_warmup(model):
    warm_img = torch.rand(1, 3, 224, 224).to(device)
    for i in range(2):
        inference(model, warm_img)
    print("warm up over")
    return 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    model = model_init(model_path="./weights/best_model.pth")
    net_warmup(model)

    while True:
        user_input = input("start\n")
        if user_input == "1":
            detection(
                img_path="/home/abelxiaoxing/work/datas/package/package_test/train/EmptyBag/1031142320489_botton.bmp",
                picture_id="01111",
                model=model,
            )
        else:
            print("输入错误，请重新输入！")
