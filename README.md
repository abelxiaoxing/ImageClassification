## 基于pytorch的深度学习图像分类框架使用简介
1. 下载好数据集，代码中默认使用的是迷你猫狗分类数据集，下载地址: [https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz),
如果下载不了的话可以通过百度云链接下载: 链接: [迷你猫狗分类数据集](https://pan.baidu.com/s/16SPmrN_PUUTWQuxtRXZmrA?pwd=abel) 提取码: abel 
### train.py
1. 在`train.py`脚本中将`--data_path`设置成数据集文件夹绝对路径
2. `--model`参数可设置为timm库中任意的模型，且都支持预训练权重(下载预训练权重需要科学上网)
3. `--train_split_rato`参数为0,则为手动划分数据集模式，数据集路径内为`train`和`val`，若为0到1的浮点数，数据集内直接为样本标签，所划分的训练集比例为该值。
### todo
1. 适配目标检测
2. 适配语义分割
3. 增加权重转换
4. 增加模型可视化
5. 增加剪枝、量化、蒸馏等操作
