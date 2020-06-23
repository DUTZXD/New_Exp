import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import kornia
import tv_loss
import my_loss
# from scipy.ndimage import gaussian_filter
# import Gaussian
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import utils as vutils
from ssim import SSIM
from mydataset import MyData

batch_size = 16
epoches = 1000
lr = 0.001


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

    def make_layer(self, inChannel, outChannel, kernel):
        layer = [ResidualBlock(inChannel, outChannel, kernel_size=kernel, stride=1, padding=int((kernel - 1) / 2)),
                 ResidualBlock(outChannel, outChannel, kernel_size=kernel, stride=1, padding=int((kernel - 1) / 2)),
                 ResidualBlock(outChannel, outChannel, kernel_size=kernel, stride=1, padding=int((kernel - 1) / 2))]
        return nn.Sequential(*layer)

    def forward(self, x):
        # 先进行高斯滤波及下采样
        img0 = kornia.gaussian_blur2d(x, (5, 5), (1.1, 1.1))

        img1 = F.interpolate(img0, scale_factor=0.5)
        img1 = kornia.gaussian_blur2d(img1, (3, 3), (0.8, 0.8))

        img2 = F.interpolate(img1, scale_factor=0.5)
        img2 = kornia.gaussian_blur2d(img2, (1, 1), (0.5, 0.5))

        x0 = self.layer0(img0)
        x1 = self.layer1(img1)
        x2 = self.layer2(img2)
        # save_image(x2, './tmp3/illu2.png')
        # save_image(x1, './tmp3/illu1.png')
        # save_image(x0, './tmp3/illu0.png')

        x0_log = torch.log(x0 + 1e-10)
        x1_log = torch.log(x1 + 1e-10)
        x2_log = torch.log(x2 + 1e-10)

        x2_log = F.interpolate(x2_log, size=[x0_log.shape[2], x0_log.shape[3]], mode='nearest')
        x1_log = F.interpolate(x1_log, size=[x0_log.shape[2], x0_log.shape[3]], mode='nearest')

        illumination_log = (x0_log + x1_log + x2_log)/3.0
        return illumination_log


# 保存结果
def save_image(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)


# 训练
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion1 = nn.MSELoss()
    criterion2 = tv_loss.TVLoss()
    criterion3 = my_loss.MyLoss()
    running_loss = 0.0
    for batch_index, train_data in enumerate(train_loader):
        inputs, labels = train_data
        inputs, labels = inputs.to(device), labels.to(device)
        input_log = torch.log(inputs + 1e-10)
        illus = model(inputs)
        outputs = torch.exp(input_log - illus)
        # loss = criterion1(outputs, labels) + criterion2(illus) + criterion3(outputs, labels)
        loss = criterion1(outputs, labels) + criterion2(illus)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('batch_index:', batch_index,
              'Epoch: %d Current train loss: %4f' % (epoch, running_loss / (batch_index + 1)))
    torch.save(model.state_dict(), 'net_params_v2.pkl')
    print('loss: ', running_loss)
    with open('loss_v2.txt', 'a') as f:
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
            input_log = torch.log(inputs + 1e-10)
            illus = model(inputs)
            outputs = torch.exp(input_log - illus)
            # print(input_log)
            save_image(outputs, './tmp3/' + str(index) + '.png')
            test_loss += criterion(outputs, labels).item()
            # if(index==0):
            #     break
    test_loss /= len(test_loader.dataset)
    print('Epoch: %d Current test loss: ', epoch, test_loss)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load('net_params_v2.pkl'))

    train_data = MyData('./train.txt', transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = MyData('./test.txt', transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epoches):
        train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader, 1)


if __name__ == '__main__':
    main()
