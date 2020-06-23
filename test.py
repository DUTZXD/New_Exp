import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from ssim import SSIM
from mydataset import MyData
import numpy
import math


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) / 3.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = MyData('./cur_result.txt', transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    ssim_sum = 0.0
    criterion1 = SSIM()
    # f = open("SSIM&PSNR.txt", 'w')
    for batch_index, train_data in enumerate(test_loader):
        results, labels = train_data
        results, labels = results.to(device), labels.to(device)
        ssim = criterion1(results, labels)
        ssim_sum += ssim.item()
        # ps = psnr(results, labels)
        # line = str(ssim.item()) + " ----- " + str(ps)
        # f.write(line + "\n")
    # f.close()
    print(ssim_sum/487)


if __name__ == '__main__':
    main()
