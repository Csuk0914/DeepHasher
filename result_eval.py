
# Evaluate training result on test dataset
# import numpy as np
# import pydicom
# import os
# import csv
#import pickle
# import matplotlib.pyplot as plt
#from scipy.interpolate import RegularGridInterpolator
#from PIL import Image
from hashingNet import HashingNet, SliceDataSet, configs

import torch
import torch.nn as nn
from torch.autograd import Variable
#from torch import optim
from torch.utils.data import DataLoader


weights_dir = "./params.pth.tar"
net = HashingNet().cuda()
net.load_state_dict(torch.load(weights_dir))
net.eval()
loss_fn = nn.MSELoss()
slice_test = SliceDataSet(data_dir='../data_test')
test_loader = DataLoader(slice_test, batch_size=configs['batch_train'], shuffle=False, num_workers=configs['num_workers'])
total_loss = 0.0
for batch_idx, batch_sample in enumerate(test_loader):
    img = batch_sample['img']
    label = batch_sample['label']
    img, y = Variable(img).cuda(), Variable(label).cuda()
    y_pred = net(img)
    mse_loss = loss_fn(y_pred, y)

    if batch_idx % (len(slice_test) / configs['batch_test'] / 5) == 0:
        print("Batch %d Loss %f" % (batch_idx, mse_loss.item()))
        total_loss += mse_loss.item()
mean_loss = total_loss / float(len(slice_test)/configs['batch_test'])
print("Average BCE loss on training data is: ", mean_loss)