import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader
import random
from PIL import Image

transform=transforms.Compose([
    transforms.Resize(100),
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])

class Siamese_MNIST(Dataset):
    def __init__(self,mnist_dataset):
        self.data=mnist_dataset
        self.transform=transform
    def __getitem__(self, item):
        img1,label1=self.data[item]
        should_match=random.randint(0,1)
        while True:
            img2,label2=random.choice(self.data)
            if (label1==label2 and should_match==1) or (label1!=label2 and should_match==0):
                break
        return self.transform(img1),self.transform(img2),torch.tensor([int(label1==label2)],dtype=torch.float)
    def __len__(self):
        return len(self.data)

class SiameseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc=nn.Sequential(
            nn.Linear(32*23*23,128),
            nn.ReLU(),
            nn.Linear(128,64)
        )
    def forward_once(self,x):
        x=self.conv(x)
        x=torch.flatten(x,1)
        return self.fc(x)

    def forward(self,x1,x2):
        out1=self.forward_once(x1)
        out2=self.forward_once(x2)
        return out1,out2

class ContrastiveLoss(nn.Module):
    def __init__(self,margin=1.0):
        super().__init__()
        self.margin=margin
    def forward(self,out1,out2,label):
        label=label.view(-1)
        distance=F.pairwise_distance(out1,out2)
        loss=distance.pow(2)*label + (1-label)*F.relu(self.margin-distance).pow(2)
        return loss.mean()

mnist=datasets.MNIST(root="./data",train=True,download=True)
siamese_datasets=Siamese_MNIST(mnist)
loader=DataLoader(siamese_datasets,batch_size=64,shuffle=True)

model=SiameseCNN()
criterion=ContrastiveLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(5):
    for img1,img2,label in loader:
        out1,out2=model(img1,img2)
        loss=criterion(out1,out2,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss is {loss.item():.4f}")

def predict_similarity(model,img_path1,img_path2,transform,threshold=0.5):
    model.eval()
    img11=Image.open(img_path1).convert("L")
    img21=Image.open(img_path2).convert("L")
    img11=transform(img11).unsqueeze(0)
    img21=transform(img21).unsqueeze(0)

    with torch.no_grad():
        out11,out21=model(img11,img21)
        distance=F.pairwise_distance(out11,out21)
        is_same=distance.item()<threshold
        print(f"distance : {distance.item():.4f}")
        print("prediction:same" if is_same else "prediction:different")
        return is_same

#in blank spaces write the image paths
predict_similarity(model,"","",transform)