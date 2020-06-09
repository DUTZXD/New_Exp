import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import kornia
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import utils as vutils

batch_size = 16
epoches = 1000
lr = 0.001


# 定义残差块结构
class ResidualBlock(nn.Module):
    def __init__(self, inChannel, outChannel, kernel_size, stride, padding, name=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size, stride, padding),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(True),
            nn.Conv2d(outChannel, outChannel, kernel_size, stride, padding),
            nn.BatchNorm2d(outChannel),
        )
        self.name = name

    def forward(self, x):
        out = self.left(x)
        residual = x
        out += residual
        # if self.name == 'layer0block3' or self.name == 'layer1block3' or self.name == 'layer2block3':
        #     save_image(out, './tmp/'+self.name+'.png')
        return F.relu(out)


# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer0 = self.make_layer(3, 3, 'layer0')
        self.layer1 = self.make_layer(3, 3, 'layer1')
        self.layer2 = self.make_layer(3, 3, 'layer2')

    def make_layer(self, inChannel, outChannel, name):
        layer = [ResidualBlock(inChannel, outChannel, kernel_size=3, stride=1, padding=1, name=name+'block1'),
                 ResidualBlock(outChannel, outChannel, kernel_size=5, stride=1, padding=2, name=name+'block2'),
                 ResidualBlock(outChannel, outChannel, kernel_size=7, stride=1, padding=3, name=name+'block3'),
                 ResidualBlock(outChannel, outChannel, kernel_size=5, stride=1, padding=2, name=name+'block4'),
                 ResidualBlock(outChannel, outChannel, kernel_size=3, stride=1, padding=1, name=name+'block5')]
        return nn.Sequential(*layer)

    def forward(self, x):
        # 先进行高斯滤波及下采样
        img0 = kornia.gaussian_blur2d(x, (7, 7), (1.4, 1.4))

        img1 = F.interpolate(img0, scale_factor=0.5)
        img1 = kornia.gaussian_blur2d(img1, (5, 5), (1.1, 1.1))

        img2 = F.interpolate(img1, scale_factor=0.5)
        img2 = kornia.gaussian_blur2d(img2, (3, 3), (0.8, 0.8))

        input_log = torch.log(x + 1e-10)
        img0_log = -torch.log(img0 + 1e-10)
        img1_log = -torch.log(img1 + 1e-10)
        img2_log = -torch.log(img2 + 1e-10)

        x2 = -self.layer2(img2_log)
        # save_image(-x2, './tmp/x2.png')
        x1 = -self.layer1(img1_log)
        # save_image(-x1, './tmp/x1.png')
        x0 = -self.layer0(img0_log)
        # save_image(-x0, './tmp/x0.png')

        x2 = F.interpolate(x2, size=[img0_log.shape[2], img0_log.shape[3]], mode='nearest')
        x1 = F.interpolate(x1, size=[img0_log.shape[2], img0_log.shape[3]], mode='nearest')
        output = torch.exp(input_log - x0 - x1 - x2)
        return output


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
        img, label = my_transform(img, label)
        img = transforms.ToPILImage()(img).convert('RGB')
        label = transforms.ToPILImage()(label).convert('RGB')
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
    criterion = nn.MSELoss()
    running_loss = 0.0
    for batch_index, train_data in enumerate(train_loader):
        inputs, labels = train_data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('batch_index:', batch_index, 'Epoch: %d Current train loss: %4f' % (epoch, running_loss / (batch_index + 1)))
    torch.save(model.state_dict(), 'net_params2.pkl')
    print('loss: ', float(running_loss))
    with open('loss2.txt', 'a') as f:
        f.write(str(float(running_loss)) + "\n")
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
            outputs = model(inputs)
            save_image(outputs, './result_v2/' + str(index) + '.png')
            test_loss += criterion(outputs, labels).item()
    test_loss /= len(test_loader.dataset)
    print('Epoch: %d Current test loss: %4f', epoch, test_loss)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load('net_params2.pkl'))

    train_data = MyData('./train.txt', transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = MyData('./test.txt', transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epoches):
        train(model, device, train_loader, optimizer, epoch)
    # test(model, device, test_loader, 1)


if __name__ == '__main__':
    main()
