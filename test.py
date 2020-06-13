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

batch_size = 16
epoches = 800
lr = 0.0003


# 定义残差块结构
class ResidualBlock(nn.Module):
    def __init__(self, inChannel, outChannel, kernel_size, stride, padding, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size, stride, padding),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(True),
            nn.Conv2d(outChannel, outChannel, kernel_size, stride, padding),
            nn.BatchNorm2d(outChannel),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        # residual = x if self.right is None else self.right(x)
        # out += residual
        return F.relu(out)


# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer0 = self.make_layer(3, 3, 5)
        self.layer1 = self.make_layer(3, 3, 3)
        self.layer2 = self.make_layer(3, 3, 1)
        # self.smoothing0 = Gaussian.GaussianSmoothing(3, 7, 1.3)
        # self.smoothing1 = Gaussian.GaussianSmoothing(3, 5, 1.1)
        # self.smoothing2 = Gaussian.GaussianSmoothing(3, 3, 0.8)
        # self.gaussian = Gaussian.GaussianBlurConv()

    def make_layer(self, inChannel, outChannel, kernel):
        layer = [ResidualBlock(inChannel, outChannel, kernel_size=kernel, stride=1, padding=int((kernel-1)/2)),
                 ResidualBlock(outChannel, outChannel, kernel_size=kernel, stride=1, padding=int((kernel-1)/2)),
                 ResidualBlock(outChannel, outChannel, kernel_size=kernel, stride=1, padding=int((kernel-1)/2))]
        return nn.Sequential(*layer)

    def forward(self, x):
        # 先进行高斯滤波及下采样
        # img0 = self.gaussian(x)
        # img0 = self.smoothing0(x)
        img0 = kornia.gaussian_blur2d(x, (5, 5), (1.1, 1.1))

        img1 = F.interpolate(img0, scale_factor=0.5)
        # img1 = self.gaussian(img1)
        # img1 = g_difference(img1, 5)
        img1 = kornia.gaussian_blur2d(img1, (3, 3), (0.8, 0.8))

        img2 = F.interpolate(img1, scale_factor=0.5)
        # img2 = self.gaussian(img2)
        # img2 = g_difference(img2, 3)
        img2 = kornia.gaussian_blur2d(img2, (1, 1), (0.5, 0.5))

        img0_log = -torch.log(img0 + 0.0001)
        img1_log = -torch.log(img1 + 0.0001)
        img2_log = -torch.log(img2 + 0.0001)

        x2 = -self.layer2(img2_log)
        # save_image(-x2, './tmp2/x2.png')
        x1 = -self.layer1(img1_log)
        # save_image(-x1, './tmp2/x1.png')
        x0 = -self.layer0(img0_log)
        # save_image(-x0, './tmp2，/x0.png')
        x2 = F.interpolate(x2, size=[img0_log.shape[2], img0_log.shape[3]], mode='nearest')
        x1 = F.interpolate(x1, size=[img0_log.shape[2], img0_log.shape[3]], mode='nearest')
        illumination = x0 + x1 + x2
        return illumination


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


# 保存结果
def save_image(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)


# 同时对input,label做随机裁剪
def my_transform(input_img, label):
    i, j, h, w = transforms.RandomCrop.get_params(input_img, (64, 64))
    image = tf.crop(input_img, i, j, h, w)
    label = tf.crop(label, i, j, h, w)
    image = tf.to_tensor(image)
    label = tf.to_tensor(label)
    return image, label


# 训练
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion1 = nn.MSELoss()
    # criterion1 = SSIM()
    criterion2 = tv_loss.TVLoss()
    criterion3 = my_loss.MyLoss()
    running_loss = 0.0
    for batch_index, train_data in enumerate(train_loader):
        inputs, labels = train_data
        inputs, labels = inputs.to(device), labels.to(device)
        input_log = torch.log(inputs + 0.0001)
        illus = model(inputs)
        outputs = torch.exp(input_log - illus)
        loss = criterion1(outputs, labels) + criterion2(illus) + criterion3(outputs, labels)
        running_loss += float(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('batch_index:', batch_index, 'Epoch: %d Current train loss: %4f' % (epoch, running_loss/(batch_index+1)))
    torch.save(model.state_dict(), 'net_params5.pkl')
    print('loss: ', float(running_loss))
    with open('loss.txt', 'a') as f:
        f.write(str(running_loss) + "\n")
    print('第', epoch, '轮训练结束，网络参数更新')


# 测试
def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0.0
    criterion = nn.L1Loss()
    with torch.no_grad():
        for index, test_data in enumerate(test_loader):
            inputs, labels = test_data
            inputs, labels = inputs.to(device), labels.to(device)
            input_log = torch.log(inputs + 0.0001)
            illus = model(inputs)
            outputs = torch.exp(input_log - illus)
            save_image(outputs, './tmp2/' + str(index) + '.png')
            test_loss += criterion(outputs, labels).item()
    test_loss /= len(test_loader.dataset)
    print('Epoch: %d Current test loss: ', epoch, test_loss)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load('net_params5.pkl'))

    train_data = MyData('./train.txt', transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = MyData('./test.txt', transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    # for epoch in range(epoches):
    #     train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader, 1)


if __name__ == '__main__':
    main()
