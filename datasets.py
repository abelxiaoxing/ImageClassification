import torch
import random
import os
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from abel_augmentations import AbelAugment
import json
from pathlib import Path
from torch.utils.data import random_split

# 分割数据集为训练集和验证集
def split_dataset(root, train_ratio=0.9):
    """将数据集划分为训练集和验证集。"""
    dataset = datasets.ImageFolder(root)  # 读取数据集
    num_train = int(len(dataset) * train_ratio)  # 计算训练集的大小
    num_val = len(dataset) - num_train  # 计算验证集的大小
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])  # 随机划分数据集
    return train_dataset, val_dataset, dataset.class_to_idx  # 返回训练集、验证集和类别索引

# 构建数据集
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)  # 构建数据转换

    print("Transform = ")
    if isinstance(transform, tuple):  # 打印转换信息
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.train_split_rato == 0:  # 如果数据集是手动设置
        root = os.path.join(args.data_path, "train" if is_train else "val")  # 根据训练或验证选择路径
        dataset = datasets.ImageFolder(root, transform=transform)  # 创建数据集
        class_indices = dataset.class_to_idx  # 获取类别索引
        json_str = json.dumps(
            dict((val, key) for key, val in class_indices.items()), indent=4
        )
        with open(
            Path(str(args.output_dir) + "/class_indices.json"), "w"
        ) as f:
            f.write(json_str)  # 将类别索引保存为JSON文件
        num_classes = args.num_classes  # 获取类别数量
        assert len(dataset.class_to_idx) == num_classes  # 确认类别数量一致
    else:  # 如果数据集是自动生成
        dataset_root = args.data_path
        train_ratio = args.train_split_rato
        train_dataset, val_dataset, class_indices = split_dataset(
            dataset_root, train_ratio
        )

        if is_train:
            dataset = train_dataset
        else:
            dataset = val_dataset

        dataset.dataset.transform = transform  # 应用转换到原始数据集

        json_str = json.dumps(
            dict((val, key) for key, val in class_indices.items()), indent=4
        )
        with open(
            Path(str(args.output_dir) + "/class_indices.json"), "w"
        ) as f:
            f.write(json_str)  # 将类别索引保存为JSON文件

        num_classes = args.num_classes  # 获取类别数量
        assert len(class_indices) == num_classes  # 确认类别数量一致
    print("Number of the class = %d" % num_classes)  # 打印类别数量

    return dataset, num_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        transform = []
        transform = create_transform(
            input_size=args.input_size,
            scale=(1.0, 1.0),
            ratio=(1.0, 1.0),
            is_training=True,
            vflip=0.5,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        # transform.transforms.insert(0,MyAugmentation(1))
        return transform

    t = []
    if resize_im:
        if args.input_size >= 384:
            t.append(
                transforms.Resize(
                    (args.input_size, args.input_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is True:
                t.append(transforms.CenterCrop(args.input_size))
            size = args.input_size
            t.append(
                transforms.Resize(
                    [size, size], interpolation=transforms.InterpolationMode.BICUBIC
                ),
            )
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class MyAugmentation(torch.nn.Module):
    def __init__(self, size=2, prob=0.5):
        super().__init__()
        self.size = random.randint(0, size)
        self.size = size
        self.prob = prob
        self.transform_list = [
            transforms.RandomRotation(90, expand=True),
            AbelAugment(1),
            transforms.GaussianBlur(kernel_size=random.choice([1, 3, 5])),
            transforms.RandomPerspective(
                distortion_scale=random.uniform(0, 0.4), p=self.prob, fill=0
            ),
            transforms.ElasticTransform(alpha=random.uniform(0, 10)),
            transforms.RandomVerticalFlip(),
            transforms.RandomInvert(p=self.prob),
        ]

    def forward(self, x):
        for _ in range(self.size):
            transform = transforms.RandomApply(self.transform_list, p=self.prob)
            x = transform(x)
        return x
