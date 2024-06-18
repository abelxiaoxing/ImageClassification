 
import torch
import torch.nn.functional as F
from model.Mynet import Mynetinit
from dataset.datasets import DogCat
import os
from tqdm import tqdm
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Mynetinit()

model.load_state_dict(torch.load("./pth/best.pth"))
model = model.to(device)

num_classes = 5 

def inference(model, device, test_loader):
    class_cnt = defaultdict(int)
    tp = defaultdict(int)
    fp = defaultdict(int)
    tn = defaultdict(int)
    fn = defaultdict(int)

    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_class = output.argmax(dim=1)
            for cls in range(num_classes):
                class_cnt[cls] += (target == cls).sum().item()
                tp[cls] += ((output_class == cls) & (target == cls)).sum().item()
                fp[cls] += ((output_class == cls) & (target != cls)).sum().item()
                tn[cls] += ((output_class != cls) & (target != cls)).sum().item()
                fn[cls] += ((output_class != cls) & (target == cls)).sum().item()
            correct += output_class.eq(target.view_as(output_class)).sum().item()

    accuracy = 100 * correct / len(DogCat.test_data)
    print('Test accuracy: %.4f' % accuracy)
    
    for cls in range(num_classes):
        if tp[cls] + fp[cls] > 0:
            precision = tp[cls] / (tp[cls] + fp[cls])
        else:
            precision = 0
            print(tp[cls],fp[cls])
        if tp[cls] + fn[cls] > 0:
            recall = tp[cls] / (tp[cls] + fn[cls])
        else:
            precision = 0
            print(tp[cls],fn[cls])
        print('Class %d precision: %.4f recall: %.4f' % (cls, precision, recall))

test_loader = torch.utils.data.DataLoader(DogCat.test_data, batch_size=32, shuffle=False, num_workers=4)
inference(model, device, test_loader)
