import argparse
import os
import torch
import cv2 as cv
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
'''
车牌识别应用场景中一般有两部分组成：1.图像切割部分（定位出车牌所在区域）；2.图像识别部分。本实验为第二部分
车牌识别属于监督学习，
将汉字部分分离，单独训练；将整个车牌一起识别
python3.4+、tersorflow 

'''
#load train_data 加载训练数据集
PICS_PATH = "../data/train"

# provice2index：31 个省（直辖市）对应的缩影
INDEX_PROVINCE = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10,
                  "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20,
                  "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30}
#index_letter: 除省之外的字符
INDEX_LETTER = {"0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47,"H": 48, "J": 49, "K": 50, "L": 51, "M": 52,
                "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60, "W": 61, "X": 62, "Y": 63, "Z": 64}

#训练数据加载类
class CarPlateLoader(Dataset):
    def __init__(self, pics):
        self.pics = pics

    def __len__(self):
        return len(self.pics)

    def __getitem__(self, item):
        img = cv.imread(PICS_PATH + "/" + self.pics[item])
        r, g, b = cv.split(img)
        numpy_array = np.array([r, g, b])
        label = self.pics[item][0:7]

        img_tensor = torch.from_numpy(numpy_array)
        img_tensor = img_tensor.float()

        label_tensor = torch.zeros(238)
        label_tensor = label_tensor.float()
        label_tensor[int(INDEX_PROVINCE[label[0]])] = float(1)
        for i in range(6):
            index = int(INDEX_LETTER[label[i + 1]])
            index = index - 31
            label_tensor[34 + i * 34 + index] = float(1)
        return {"img":img_tensor, "label":label_tensor}

#卷积神经网络 4C + 3L,dropout来去除过拟合
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.conv4 = nn.Conv2d(32, 10, 1)
        self.fc1 = nn.Linear(28 * 7 * 10, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 238)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.3)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.avg_pool2d(x, (2, 2))
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, 28 * 7 * 10)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = x.view(-1, 7, 34)
        x = F.softmax(x, dim=2)
        x = x.view(-1, 238)
        return x



#传参：迭代周期、学习率和batch 大小，默认值
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('epoes', type=int, default=30, help='train epoes')
    parser.add_argument('lr', type=float, default=0.0001, help='train lr')
    parser.add_argument('batchSize', type=int, default=50, help='train batch_size')

    return parser.parse_args()


def main(args):
    pics = os.listdir(PICS_PATH)
    data_set = CarPlateLoader(pics)
    data_loader = DataLoader(data_set, batch_size=args.batchSize, shuffle=True, num_workers=8)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.9)

    for i in range(args.epoes):
        for i_batch, sample_batched in enumerate(data_loader):
            optimizer.zero_grad()
            img_tensor = sample_batched["img"].to(device)
            label_tensor = sample_batched["label"].to(device)
            output = model(img_tensor)
            loss = F.mse_loss(output, label_tensor)
            loss.backward()
            optimizer.step()
            if i_batch % 10 == 0:
                print(i, i_batch, "loss="+str(loss.cpu().item()), "lr="+str(scheduler.get_lr()))
        scheduler.step()

    torch.save(model.state_dict(), "../model/model.pt")


if __name__ == '__main__':
    main(parse_args())
