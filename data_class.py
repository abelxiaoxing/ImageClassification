#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,shutil
import random

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print(srcfile)
        print("src not exist!")
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件

if __name__ == '__main__':
    test_rate = 0.1  # 训练集和测试集的比例为9:1
    file_path = r"./flower_photos"
    train_folder = "train"
    test_folder = "test"
    # categories = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    train_path = os.path.join(file_path, train_folder)
    categories = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]

    for category in categories:
        train_category_path = os.path.join(file_path, train_folder, category)
        test_category_path = os.path.join(file_path, test_folder, category)
        
        # Get all image files in the train category folder
        img_files = [f for f in os.listdir(train_category_path) if os.path.isfile(os.path.join(train_category_path, f))]
        
        img_num = len(img_files)
        test_num = int(img_num * test_rate)
        
        # Randomly select test_num images to move to the test set
        test_files = random.sample(img_files, test_num)
        
        for img_file in test_files:
            srcfile = os.path.join(train_category_path, img_file)
            dstfile = os.path.join(test_category_path, img_file)
            mymovefile(srcfile, dstfile)

    print("Data split complete.")
