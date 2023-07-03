import os
import numpy as np

import torch
import torch.nn as nn

from datetime import datetime

## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/%s_model_epoch%d.pth" % (ckpt_dir, current_time, epoch))

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    # ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print(ckpt_lst[0])
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[0]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[0].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch
