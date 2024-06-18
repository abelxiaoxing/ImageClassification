## 基于pytorch的深度学习图像分类框架使用简介
1. 下载好数据集，代码中默认使用的是花分类数据集，下载地址: [https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz),
如果下载不了的话可以通过百度云链接下载: 链接: https://pan.baidu.com/s/1FJhWJVY7yEIRyEwcWGdllg?pwd=abel 提取码: abel 
### train.py
+ 模型的训练代码
### val.py
+ 模型的预测代码
### data_class.py
+ 数据集自动分割成train和test
+ 将原本的数据集的类别文件夹所在的位置新建train文件夹，并将这些类别文件夹放入train文件夹中，运行data_class即可自动分割成train和test数据集
+ 调整test_rate决定划分的比例
### view.py
+ 将data_class分割好的数据放入当前目录下的`datas`内，即可随机选取部分数据集进行可视化并展现标签
### ViewParameters.py
+ 查看模型权重大小
