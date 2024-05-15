import torch
from timm.models import create_model
model = create_model(model_name='eva02_tiny_patch14_224.mim_in22k',num_classes=2)
model.load_state_dict(torch.load("best.pt", map_location="cuda")['model'])
model.cuda()
model.eval()
img = torch.rand(1, 3, 224, 224).cuda()
# 导出模型为ONNX
torch.onnx.export(model,                      # 导出的模型
                  img,                        # 示例输入张量
                  'weights/best_model.onnx',  # 保存的文件路径
                  input_names=['input'],       # 输入张量的名称
                  output_names=['output'])
