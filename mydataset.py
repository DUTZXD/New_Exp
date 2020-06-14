import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import kornia
import tv_loss
import my_loss
from scipy.ndimage import gaussian_filter
import Gaussian
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import utils as vutils
from ssim import SSIM


# 自定义数据集
class MyData(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split('***', 1)
            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        fh.close()

    def __getitem__(self, index):
        img, label = self.imgs[index]
        img = Image.open(img).convert('RGB')
        label = Image.open(label).convert('RGB')
        # img, label = my_transform(img, label)
        # img = transforms.ToPILImage()(img).convert('RGB')
        # label = transforms.ToPILImage()(label).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)
        return img, label

    def __len__(self):
        return len(self.imgs)


# 同时对input,label做随机裁剪
def my_transform(input_img, label):
    i, j, h, w = transforms.RandomCrop.get_params(input_img, (64, 64))
    image = tf.crop(input_img, i, j, h, w)
    label = tf.crop(label, i, j, h, w)
    image = tf.to_tensor(image)
    label = tf.to_tensor(label)
    return image, label