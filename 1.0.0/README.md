## 基于pytorch的深度学习图像分类模型训练使用简介
### 下载数据集
+ 代码中默认使用的是花分类数据集，下载地址: [https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz),
如果下载不了的话可以通过百度云链接下载: 链接: https://pan.baidu.com/s/1FJhWJVY7yEIRyEwcWGdllg?pwd=abel 提取码: abel 
### 训练方法
1. 将数据集手动分割成train和val,分别放在当前目录下的`"./flower_photos/train/"`和`"./flower_photos/val/"`文件夹下。
2. 若需要使用自己的数据集，则修改
        `torch.nn.Linear(256,5)`中的5，改成自己数据集所需要预测的类别数
3. 运行`ori.py`文件即可开始训练。
