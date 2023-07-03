## 라이브러리 추가하기
import argparse

import os
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from model.ms_model import ResUNet_MS
from utils.dataset import *
from utils.util import *
from utils.postprocessing import *
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import medpy.metric.binary as bin
from datetime import datetime
import pandas as pd
from loss.loss import DC_and_BCE_loss

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


## Arg_Parser 
parser = argparse.ArgumentParser()

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--cuda_devices", default="0", type=str,
                    help="String of cuda device indexes to be used. Indexes must be separated by a comma.")
parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")
parser.add_argument("--fp16", action="store_true", help="run with mixed precision")

args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue
fp16 = args.fp16

device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))


net = ResUNet_MS().to(device) # ResUNet_MS
net = nn.DataParallel(module=net).to(device)

if mode == 'train':

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), train_transform=True)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), train_transform=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)
    # print(len(loader_val))
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    print('train dataset num: ',(num_data_train, num_data_val))

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    batch_size = 1
    dataset_test = Dataset(data_dir=data_dir, train_transform=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)
    
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)
    print("test dataset num: ", num_data_test)

fn_loss = DC_and_BCE_loss({}, {}).to(device)

optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.99, nesterov=True)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))
amp_grad_scaler = GradScaler()

st_epoch = 0
best_metric = -1
val_interval = 5

os.makedirs(result_dir, exist_ok=True)
path = os.path.join(result_dir, 'analysis.xlsx')
writer = pd.ExcelWriter(path, engine = 'openpyxl')

# TRAIN MODE
if mode == 'train':
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
    with trange(st_epoch +1, num_epoch + 1) as tbar:

        for epoch in tbar:
            net.train()
            loss_arr = []
            dice_arr = []

            for batch, data in enumerate(loader_train, 1):

                input = data[0].to(device, dtype=torch.float32)
                label = data[1].to(device, dtype=torch.float32)
                optim.zero_grad()
                
                if fp16:
                    with autocast():
                        output = net(input)
                        loss = fn_loss(output, label)

                        amp_grad_scaler.scale(loss).backward()
                        amp_grad_scaler.step(optim)
                        amp_grad_scaler.update() # neccessary if autocast
                else:
                    output = net(input)
                    loss = fn_loss(output, label)
                    loss.backward()
                    optim.step()

                loss_arr += [loss.item()]

                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                dice = bin.dc(output, label)
                dice_arr += [dice]

                tbar.set_description('Epoch {} Loss {:.4f} Dice {}'.format(epoch, np.mean(loss_arr), np.round(dice, 4)))

                writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

            writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
            writer_train.add_scalar('dice', np.mean(dice_arr), epoch)

            with torch.no_grad():
                net.eval()
                loss_arr = []
                dice_arr = []

                for batch, data in enumerate(loader_val, 1):
                    # forward pass
                    tbar.set_description('Testing {}/{}'.format(batch, len(loader_val)))
                    input = data[0].to(device, dtype=torch.float32)
                    label = data[1].to(device, dtype=torch.float32)
                    
                    if fp16:
                        with autocast():
                            output = net(input)
                            loss = fn_loss(output, label)
                    else:
                        output = net(input)
                        loss = fn_loss(output, label)
                    
                    loss_arr += [loss.item()]

                    label = fn_tonumpy(label)
                    input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                    output = fn_tonumpy(fn_class(output))
                    
                    #compute val dice
                    dice = bin.dc(output, label)
                    dice_arr += [dice]
                    tbar.set_description('Validation Dice = {}'.format(np.round(dice,4)))
                    

                    writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                    writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                    writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

            writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
            writer_val.add_scalar('dice', np.mean(dice_arr), epoch)

            if epoch % 100 == 0:
                save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

# TEST MODE

else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim, mode=mode)

    if not os.path.exists(os.path.join(result_dir, 'png')):
        os.mkdir(os.path.join(result_dir, 'png'))
        os.mkdir(os.path.join(result_dir, 'numpy'))
    c = 0
    p = 1
    with torch.no_grad():
        net.eval()
        for batch, data in enumerate(loader_test, 1):

            input = data.to(device, dtype=torch.float)
            if fp16:
                with autocast():
                    output = net(input)
            else:
                output = net(input)
            
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            for j in range(input.shape[0]):
                
                id = batch_size * (batch - 1) + j

                plt.imsave(os.path.join(result_dir, 'png', f'input_{id}.png'), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', f'output_{id}.png'), output[j].squeeze(), cmap='gray')

                # reread output due to cv2 type
                o = os.path.join(result_dir, 'png', f'output_{id}.png')
                o = cv2.imread(o, 0)
                img_contour, contour, hie, pp = draw_contour(o)
                info, c_im = analysis(img_contour, contour, hie)
                df = pd.DataFrame(info)
                df_t = df.transpose()
                mean_df = np.round(df_t.mean(level=None), 2)
                sum_df = np.round(df_t.sum(level=None), 2)
                df_t = df_t.append(sum_df, ignore_index=True)
                df_t.at[len(df_t)-1, "Unnamed: 0"] = "sum"
                df_t = df_t.append(mean_df, ignore_index=True)
                df_t.at[len(df_t)-1, "Unnamed: 0"] = "mean"
                df_t.to_excel(writer, sheet_name=f'contour_{id}')
                
                plt.imsave(os.path.join(result_dir, 'png', f'contour_{id}.png'), img_contour, cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', f'pp_{id}.png'), pp, cmap='gray')

                np.save(os.path.join(result_dir, 'numpy', f'input_{id}.npy'), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', f'output_{id}.npy'), output[j].squeeze())

            writer.save()