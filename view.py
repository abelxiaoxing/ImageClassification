from model.cnn import CNN
import torch
from dataset.datasets import DogCat
import torch.nn.functional as F
import pandas as pd
import random
import matplotlib.pyplot
import PIL
import os
device=torch.device("cuda" if torch.cuda.is_available() else "cpu" )#设置训练模式 GPU or CPU
model = CNN()
model.load_state_dict(torch.load('./pth/430.pth'))#加载预训练模型
model=model.to(device)
dog_probs = []
model.eval()

with torch.no_grad():
    for data, fileid in DogCat.test_loader:
        data = data.to(device)
        output = model(data)
        output_list = F.softmax(output, dim=1)[:, 1].tolist()
        dog_probs += list(zip(list(fileid), output_list))
dog_probs.sort(key = lambda x: int(x[0]))
print(dog_probs)
idx = list(map(lambda x :x[0],dog_probs))
prob = list(map(lambda x :x[1],dog_probs))
submission = pd.DataFrame({'id':idx, 'label':prob})
# submission
submission.to_csv('result.csv',index=False)
id_list = []
class_ = {0: 'cat', 1: 'dog'}

fig, axes = matplotlib.pyplot.subplots(2, 5, figsize=(20, 12), facecolor='w')

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
        
    img_path = os.path.join('./catsdogs/test/'+random_label+ '/{}.jpg'.format(i))
    img = PIL.Image.open(img_path)
    
    ax.set_title(class_[label])
    ax.imshow(img)
