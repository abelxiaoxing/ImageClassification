## 基于pytorch的深度学习图像分类框架使用简介
1. 下载好数据集，代码中默认使用的是迷你猫狗分类数据集，下载地址:  [迷你猫狗分类数据集](https://pan.baidu.com/s/16SPmrN_PUUTWQuxtRXZmrA?pwd=abel) 提取码: abel  
### train.py
1. 在`train.py`脚本中将`--data-path`设置成解压后的`flower_photos`文件夹绝对路径
2. 去github抄所需要网络的代码，并将其放在`models`文件夹下，然后将`train.py`下导入的网络设置成你要使用的网络
3. 在`train.py`脚本中将`--weights`参数设成下载好的预训练权重路径
4. 设置好数据集的路径`--data-path`以及预训练权重的路径`--weights`就能使用`train.py`脚本开始训练了(训练过程中会自动生成`class_indices.json`文件)
5. 若需要自己划分好数据集并在所填的data_path路径后划分为train和val,则设置data_custom为True。
### val.py
1. 在`predict.py`脚本中导入和训练脚本中同样的模型，并将`model_weight_path`设置成训练好的模型权重路径(默认保存在weights文件夹下)
2. 在`predict.py`脚本中将`img_path`设置成你自己需要预测的图片绝对路径
3. 设置好权重路径`model_weight_path`和预测的图片路径`img_path`就能使用`predict.py`脚本进行预测了
### VisualDisplay.py
+ 热力图绘制
### pth2jit.py
+ 将pth权重文件转换成加密的jit文件
### pth2onnx.py
+ 将pth权重文件转换成onnx文件
### detection.py & server.py
+ 与c#进行socket通讯
### Others
+ 如果要使用自己的数据集，请按照花分类数据集的文件结构进行摆放(即一个类别对应一个文件夹)，并且将训练以及预测脚本中的`num_classes`设置成你自己数据的类别数
### todo
1. 适配目标检测
2. 适配语义分割
3. 增加权重转换
4. 增加模型可视化
5. 增加剪枝、量化、蒸馏等操作