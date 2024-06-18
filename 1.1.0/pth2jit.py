import torch
from timm.models import create_model
model = create_model(model_name='eva02_tiny_patch14_224.mim_in22k',num_classes=2)
model.load_state_dict(torch.load("pth/76_994_997.pth", map_location="cuda")['model'])
model.cuda()
model.eval()
img = torch.rand(1, 3, 224, 224).cuda()
trace_model = torch.jit.script(model, img)
torch.jit.save(trace_model, 'weights/best_model.pth')
