import datetime
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from tst import Transformer
from tst.loss import OZELoss

from src.dataset import OzeDataset
from src.utils import compute_loss, fit, Logger, kfold

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as fp  # 1、引入FontProperties
import math


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(147)

# Training parameters
# DATASET_PATH = 'C:\\Users\\14344\Desktop\\数据集\\UCRArchive_2018\\UCRArchive_2018\\ACSF1\\ACSF1_TRAIN.tsv' # 数据集路径
# train_path = 'dataset\\UCRArchive_2018\\ACSF1\\ACSF1_TRAIN.tsv'  # 数据集路径
# test_path = 'dataset\\UCRArchive_2018\\ACSF1\\ACSF1_TEST.tsv'  # 数据集路径
train_path = 'dataset\\UCRArchive_2018\\Adiac\\Adiac_TRAIN.tsv'  # 数据集路径
test_path = 'dataset\\UCRArchive_2018\\Adiac\\Adiac_TEST.tsv'  # 数据集路径
# train_path = 'dataset\\UCRArchive_2018\\BeetleFly\\BeetleFly_TRAIN.tsv'  # 数据集路径
# test_path = 'dataset\\UCRArchive_2018\\BeetleFly\\BeetleFly_TEST.tsv'  # 数据集路径
# train_path = 'dataset\\UCRArchive_2018\\BME\\BME_TRAIN.tsv'  # 数据集路径
# test_path = 'dataset\\UCRArchive_2018\\BME\\BME_TEST.tsv'  # 数据集路径
# train_path = 'dataset\\UCRArchive_2018\\CBF\\CBF_TRAIN.tsv'  # 数据集路径
# test_path = 'dataset\\UCRArchive_2018\\CBF\\CBF_TEST.tsv'  # 数据集路径
# train_path = 'dataset\\UCRArchive_2018\\BirdChicken\\BirdChicken_TRAIN.tsv'  # 数据集路径
# test_path = 'dataset\\UCRArchive_2018\\BirdChicken\\BirdChicken_TEST.tsv'  # 数据集路径
# train_path = 'dataset\\UCRArchive_2018\\ArrowHead\\ArrowHead_TRAIN.tsv'  # 数据集路径
# test_path = 'dataset\\UCRArchive_2018\\ArrowHead\\ArrowHead_TEST.tsv'  # 数据集路径

# 测试间隔 单位:epoch
data_length_p = 390  # 训练集
draw_key = 10  # 大于等于draw_key 才保存结果图

EPOCHS = 200
BATCH_SIZE = 16
test_interval = 1  # 测试间隔 单维epoch
LR = 1e-4
optimizer_p = 'Adagrad'

# Model parameters
d_channel = 512
d_model = 512  # Lattent dim
q = 8  # Query size
v = 8  # Value size
h = 6  # Number of heads
N = 6  # Number of encoder and decoder to stack
dropout = 0  # Dropout rate
pe = True  # Positional encoding
mask = True  # mask
noise = 0.01  # 扩充噪音率

d_input = 176  # From dataset
d_output = 37  # From dataset

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

reslut_figure_path = 'result_figure'  # 保存结果图像的路径
file_name = train_path.split('\\')[-1][0:train_path.split('\\')[-1].index('_')]

# Load dataset
dataset_train = OzeDataset(train_path)
dataset_test = OzeDataset(test_path)
dataloader_train = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = Data.DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# Load transformer with Adam optimizer and MSE loss function
net = Transformer(d_input, d_model, d_channel, d_output, q, v, h, N,
                  dropout=dropout, mask=mask, pe=pe, noise=noise).to(device)
if optimizer_p == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)
elif optimizer_p == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=LR)
elif optimizer_p == 'Adamax':
    optimizer = optim.Adamax(net.parameters(), lr=LR)
elif optimizer_p == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)

loss_function = OZELoss()

correct_list = []
correct_list_ontrain = []

def test(dataloader_test, flag='test_set'):
    correct = 0.0
    total = 0
    with torch.no_grad():
        for x_test, y_test in dataloader_test:
            enc_inputs, dec_inputs = x_test.to(device), y_test.to(device)
            test_outputs = net(enc_inputs)
            _, predicted = torch.max(test_outputs.data, dim=1)
            total += dec_inputs.size(0)
            correct += (predicted == dec_inputs.long()).sum().item()
        if flag == 'test_set':
            correct_list.append(round((100 * correct / total), 2))
        elif flag == 'train_set':
            correct_list_ontrain.append(round((100 * correct / total), 2))
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))


# 结果可视化 包括绘图和结果打印
def result_visualization():
    my_font = fp(fname=r"font/simsun.ttc")  # 2、设置字体路径

    # 设置风格
    # plt.style.use('ggplot')
    plt.style.use('seaborn')

    fig = plt.figure()  # 创建基础图
    ax1 = fig.add_subplot(311)  # 创建两个子图
    ax2 = fig.add_subplot(313)

    ax1.plot(loss_list)  # 添加折线
    ax2.plot(correct_list, color='red', label='on Test Dataset')
    ax2.plot(correct_list_ontrain, color='blue', label='on Train Dataset')

    # 设置坐标轴标签 和 图的标题
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel(f'epoch/{test_interval}')
    ax2.set_ylabel('correct')
    ax1.set_title('LOSS')
    ax2.set_title('CORRECT')

    plt.legend(loc='best')

    # 设置文本
    fig.text(x=0.13, y=0.4, s=f'最小loss：{min(loss_list)}' '    '
                              f'最小loss对应的epoch数:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((data_length_p / BATCH_SIZE)))}' '    '
                              f'最后一轮loss:{loss_list[-1]}' '\n'
                              f'最大correct：测试集:{max(correct_list)}% 训练集:{max(correct_list_ontrain)}%' '    '
                              f'最大correct对应的已训练epoch数:{(correct_list.index(max(correct_list)) + 1) * test_interval}' '    '
                              f'最后一轮correct：{correct_list[-1]}%' '\n'
                              f'd_model={d_model}   q={q}   v={v}   h={h}   N={N}  drop_out={dropout} dataset={file_name}'  '\n'
                              f'共耗时{round(time_cost, 2)}分钟', FontProperties=my_font)

    # 保存结果图   测试不保存图（epoch少于draw_key）
    if EPOCHS >= draw_key:
        plt.savefig(f'{reslut_figure_path}/{file_name} {max(correct_list)}% {optimizer_p} epoch={EPOCHS} batch={BATCH_SIZE} lr={LR} noise={noise} pe={pe} mask={mask} [{d_model},{q},{v},{h},{N},{dropout}].png')

    # 展示图
    plt.show()

    print('正确率列表', correct_list)

    print(f'最小loss：{min(loss_list)}\r\n'
          f'最小loss对应的epoch数:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((data_length_p / BATCH_SIZE)))}\r\n'
          f'最后一轮loss:{loss_list[-1]}\r\n')

    print(f'最大correct：测试集:{max(correct_list)}\t 训练集:{max(correct_list_ontrain)}\r\n'
          f'最correct对应的已训练epoch数:{(correct_list.index(max(correct_list)) + 1) * test_interval}\r\n'
          f'最后一轮correct:{correct_list[-1]}')

    print(f'共耗时{round(time_cost, 2)}分钟')


pbar = tqdm(total=EPOCHS)

loss_list = []

flag = False
begin_time = time()

# Prepare loss history
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=30, verbose=True)

for idx_epoch in range(EPOCHS):
    epoch_loss = []
    for idx_batch, (x, y) in enumerate(dataloader_train):
        optimizer.zero_grad()

        netout = net(x.to(device))  # [8,1460]

        loss = loss_function(y.to(device), netout)
        print('Epoch:', '%04d' % (idx_epoch + 1), 'loss =', '{:.6f}'.format(loss))
        loss_list.append(loss.item())
        epoch_loss.append(loss.item())

        loss.backward()

        optimizer.step()

    if ((idx_epoch + 1) % test_interval) == 0:
        test(dataloader_test)
        test(dataloader_train, 'train_set')
        print(f'当前最大准确率\t测试集:{max(correct_list)}%\t 训练集:{max(correct_list_ontrain)}%')

    scheduler.step(np.mean(epoch_loss))

    if pbar is not None:
        pbar.update()


end_time = time()
time_cost = round((end_time - begin_time) / 60, 2)

# 调用结果可视化
result_visualization()
