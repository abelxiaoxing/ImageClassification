import os
import argparse
import torch
from torch.utils.data import DataLoader, RandomSampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython import display
from torchvision import transforms
from my_dataset import AutoDataSet
from models.ConvNext_V2 import convnextv2_atto as create_model
from utils.base_util import (
    create_lr_scheduler,
    get_params_groups,
    train_one_epoch,
    evaluate,
    LabelSmoothingLoss,
    RASampler,
)
from utils.data_utils import read_split_data
from abel_augmentations import AbelAugment
import random
import json

class MyAugmentation(torch.nn.Module):
    def __init__(self, size=2, prob=0.5):
        super().__init__()
        self.size = random.randint(0, size)
        self.size = size
        self.prob = prob
        self.transform_list = [
            transforms.RandomRotation(90, expand=True),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.GaussianBlur(kernel_size=random.choice([1, 3, 5])),
            transforms.RandomPerspective(distortion_scale=random.uniform(0, 0.4), p=self.prob, fill=0),
            transforms.ElasticTransform(alpha=random.uniform(0, 10)),
            transforms.RandomVerticalFlip(),
            transforms.RandomInvert(p=self.prob),
        ]

    def forward(self, x):
        for _ in range(self.size):
            transform = transforms.RandomApply(self.transform_list, p=self.prob)
            x = transform(x)
        return x

def generate_class_indices(root_dir):
    classes = sorted(os.listdir(root_dir))
    class_indices = {cls: i for i, cls in enumerate(classes)}
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    return json_str

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    img_size = 224
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 20])
    print("Using {} dataloader workers every process".format(nw))

    data_transform = {
        "train": transforms.Compose(
            [
                # AbelAugment(2),
                # MyAugmentation(2),
                transforms.RandomHorizontalFlip(),
                # transforms.AugMix(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    if args.data_custom is True:

        train_images_path = []
        train_images_label = []
        for label, name in enumerate(sorted(os.listdir(args.data_path + "/train"))):
            for image_name in os.listdir(args.data_path + "/train/" + name):
                image_path = args.data_path + "/train/" + name + "/" + image_name
                train_images_path.append(image_path)
                train_images_label.append(label)

        val_images_path = []
        val_images_label = []
        for label, name in enumerate(sorted(os.listdir(args.data_path + "/val"))):
            for image_name in os.listdir(args.data_path + "/val/" + name):
                image_path = args.data_path + "/val/" + name + "/" + image_name
                val_images_path.append(image_path)
                val_images_label.append(label)
        class_indices = generate_class_indices(args.data_path + "/train")
        with open('./class_indices.json','w') as f:
            f.write(class_indices)

    else:
        (
            train_images_path,
            train_images_label,
            val_images_path,
            val_images_label,
        ) = read_split_data(args.data_path)

    train_data = AutoDataSet(
        images_path=train_images_path,
        images_class=train_images_label,
        transform=data_transform["train"],
    )
    val_data = AutoDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform["val"],
    )
    if args.repeated_augmentation is True:
        sampler_train = RASampler(train_data, num_replicas=1, rank=0, shuffle=True)
    else:
        sampler_train = RandomSampler(train_data)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=sampler_train,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_data.collate_fn,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_data.collate_fn,
    )

    model = create_model(
        num_classes=args.num_classes, in_chans=3
        ,drop_path_rate=0.1
    ).to(device)
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = torch.optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=1)

    best_acc = 0.0
    start_epoch = -1
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    learning_rate = []

    # 加载网络模型参数
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(
            args.weights
        )
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        #  print(weights_dict.keys())
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        # 黑白单通道图像加载预训练模型时，patch_embed取均值
        # for k in list(weights_dict.keys()):
        #     if "patch_embed.seq.0.c.weight" in k:
        #         weights_dict[k] = weights_dict[k].sum(dim=1).unsqueeze(1) / 3
        print(model.load_state_dict(weights_dict, strict=False))
        model.to(device)
        # 恢复训练到一半的模型
        # optimizer.load_state_dict(
        #     torch.load(args.weights, map_location=device)["optimizer"]
        # )
        # lr_scheduler.load_state_dict(
        #     torch.load(args.weights, map_location=device)["lr_scheduler"]
        # )
        # start_epoch = torch.load(args.weights, map_location=device)["epoch"]

    # 冻结除了最后一层head层以外的所有参数
    if args.freeze_layers is True:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # 开始训练
    for epoch in range(start_epoch + 1, args.epochs):
        # 打印变换后的图片
        original_image, _ = train_data[-1]
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(original_image)
        save_path = "./preprocessed_image.jpg"
        pil_image.save(save_path)
        print(f"预处理后的图像已保存至 {save_path}")
        train_loss, train_acc, lr = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
            loss_fn=LabelSmoothingLoss(0.1),
            mixup=False,
            cutmix=False,
        )
        val_loss, val_acc = evaluate(
            model=model, data_loader=val_loader, device=device, epoch=epoch
        )

        # 保存训练相关参数
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "lr_scheduler": lr_scheduler.state_dict(),
        }

        # 可视化训练情况
        train_losses.append(train_loss)
        learning_rate.append(lr)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        plt.close()
        plt.figure(figsize=(20, 6))

        plt.subplot(131)
        plt.plot(train_losses, label="train loss")
        plt.plot(val_losses, label="val loss")
        plt.legend(loc="best")
        plt.title("Losses")
        plt.xlim(0, epoch)

        plt.subplot(132)
        plt.plot(train_accs, label="train acc")
        plt.plot(val_accs, label="val acc")
        plt.legend(loc="best")
        plt.xlim(0, epoch)
        plt.title("Acc")

        plt.subplot(133)
        plt.plot(learning_rate, label="learning rate")
        plt.xlim(0, epoch)
        plt.yscale("log")
        plt.title("learning_rate")
        display.clear_output(True)
        plt.savefig("figure.png")

        torch.save(checkpoint, f"./weights/{epoch}.pth")
        if best_acc < val_acc:
            torch.save(checkpoint, "./weights/ckpt_best_%s.pth" % (str(epoch)))
            best_acc = val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeated_augmentation", type=bool, default=False)
    parser.add_argument("--data_custom", type=bool, default=True)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--data_path", type=str, default="/home/abelxiaoxing/work/datas/classification/flower_photos")
    parser.add_argument('--weights', type=str, default='./weights/convnextv2_atto_1k_224_ema.pth',help='initial weights path') 
    parser.add_argument("--freeze_layers", type=bool, default=False)
    parser.add_argument("--device", default="cuda:0", help="device id (i.e. 0 or 0,1 or cpu)")
    opt = parser.parse_args()
    main(opt)
