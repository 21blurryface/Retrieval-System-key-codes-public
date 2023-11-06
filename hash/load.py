from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import os

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, path, transform=None, loader=default_loader):

        imgs = []
        imgs.append(path)
        self.imgs = imgs
        self.transform = transform
        self.loder = loader

    def __getitem__(self,index):
        fp = self.imgs[index]
        img = self.loder(fp)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
#to ensure that the input data to the model is consistent between training and testing
def load_pic(path):
    transform_train = transforms.Compose(
        [transforms.Scale((227,227)),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = MyDataset(path=path,transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_data)
    return trainloader
