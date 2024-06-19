import torch 
import random
import os
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from abel_augmentations import AbelAugment


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        transform = []
        transform = create_transform(
            input_size=args.input_size,
            scale=(1.0, 1.0),ratio=(1.0, 1.0),
            is_training=True,vflip=0.5,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        # transform.transforms.insert(0,MyAugmentation(1))
        return transform

    t = []
    if resize_im:
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is True:
                t.append(transforms.CenterCrop(args.input_size))
            size = args.input_size
            t.append(
                transforms.Resize([size,size], interpolation=transforms.InterpolationMode.BICUBIC),  
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

