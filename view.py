from model.cnn import CNN
import torch
from dataset.datasets import DogCat
import torch.nn.functional as F
import pandas as pd
import random
import matplotlib.pyplot as plt
import PIL
import os
device=torch.device("cuda" if torch.cuda.is_available() else "cpu" )
model = CNN().to(device)
# model.load_state_dict(torch.load('./pth/430.pth'))#加载预训练模型
probs = []
model.eval()

with torch.no_grad():
    for data, target in DogCat.test_loader:
        data = data.to(device)
        output = model(data)
        output_list = F.softmax(output, dim=1)[:, 1].tolist()
        fileids = [int(os.path.splitext(os.path.basename(path))[0]) for path, _ in DogCat.test_loader.dataset.samples]
        probs += list(zip(fileids, output_list))


probs.sort(key = lambda x: int(x[0]))
idx = [x[0] for x in probs]
prob = [x[1] for x in probs]
submission = pd.DataFrame({'id':idx, 'label':prob})
# submission
submission.to_csv('result.csv',index=False)
class_ = {0: 'cat', 1: 'dog'}

fig, axes = plt.subplots(2, 5, figsize=(20, 12), facecolor='w')

for ax in axes.ravel():
    i = random.choice(submission['id'].values)
    random_label_i = random.randint(0, 1)
    if random_label_i==0:
        random_label="Cat"
    else:
        random_label="Dog"
    label = submission.loc[submission['id'] == i, 'label'].values[0]
    if label > 0.5:
        label = 1
    else:
        label = 0
        
    img_path = os.path.join(f'./datas/test/{random_label}/{i}.jpg')
    img = PIL.Image.open(img_path)
    ax.set_title(class_[label])
    ax.imshow(img)
plt.show()


