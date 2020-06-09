import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import utils as vutils
import torchvision.transforms.functional as tf

batch_size = 16
epoches = 500
lr = 0.0001
trans = transforms.Compose([
    transforms.ToTensor()])


# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5, 1, 2)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv3 = nn.Conv2d(3, 3, 1, 1, 0)

    def forward(self, x):
        R0 = torch.log(x + 0.001)
        G1 = self.conv1(R0)
        R1 = R0-G1
        G2 = self.conv1(R1)
        R2 = R1 - G2
        G3 = self.conv2(R2)
        R3 = R2 - G3
        G4 = self.conv2(R3)
        R4 = R3 - G4
        G5 = self.conv3(R4)
        R5 = R4 - G5
        G6 = self.conv3(R5)
        R6 = R5 - G6
        output = torch.exp(R6)
        output = torch.clamp(output, 0.0, 1.0)
        return output


# 自定义数据集
class MyData(Dataset):
    def __init__(self, txt_path, transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split('***', 1)
            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.transform = transform
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


def my_transform(input_img, label):
    i, j, h, w = transforms.RandomCrop.get_params(input_img, (64, 64))
    image = tf.crop(input_img, i, j, h, w)
    label = tf.crop(label, i, j, h, w)
    image = tf.to_tensor(image)
    mask = tf.to_tensor(label)
    return image, mask


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
        print('batch_index:', batch_index,
              'Epoch: %d Current train loss: %4f' % (epoch, running_loss / (batch_index + 1)))
    torch.save(model.state_dict(), 'net_params3.pkl')
    print('loss: ', running_loss)
    with open('loss_v3.txt', 'a') as f:
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
            save_image(outputs, './result_v3/' + str(index) + '.png')
            test_loss += criterion(outputs, labels).item()
    test_loss /= len(test_loader.dataset)
    print('Epoch: %d Current test loss: %4f', epoch, test_loss)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load('net_params3.pkl'))
    train_data = MyData('train.txt', transform=trans)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = MyData('test.txt', transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epoches):
        train(model, device, train_loader, optimizer, epoch)
    # test(model, device, test_loader, 1)


if __name__ == '__main__':
    main()
