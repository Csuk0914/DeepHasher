
import numpy as np
import pydicom
import os
import csv
import pickle
# import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset, DataLoader

# Setting up configuration
configs = {"batch_train": 8, \
            "batch_test": 8, \
            "epochs": 40, \
            "num_workers": 0, \
            "learning_rate": 1e-6, \
            "loss_margin": 1.0, \
            "decision_thresh": 13}

def randRot3():
    """Generate a 3D random rotation matrix.
    Returns:
        np.matrix: A 3D rotation matrix.
    """
    x1, x2, x3 = np.random.rand(3)
    R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                   [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                   [0, 0, 1]])
    v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sqrt(1 - x3)]])
    H = np.eye(3) - 2 * v * v.T
    M = -H * R
    return M


def randTrans4x4(debug=False):
    """
    Generate random 4x4 transformation
    """
    if debug:
        F = np.diag([1,1,1,1])
    else:
        F = np.zeros([4, 4])
        F[0:3, 0:3] = randRot3()
        F[2, 3] = np.random.rand(1) * 254 - 87.76
        F[3, 3] = 1.0

    return F


# Define dataset class
class SliceDataSetUnlimited(Dataset):
    """slice data set."""

    def __init__(self, data_dir='./test', data_size=1000, slice_sz=400):
        self.data_dir = data_dir
        self.data_size = data_size
        self.slice_sz = slice_sz
        self.volume_interp_func = self.getVolumeInterpFunc(data_dir)
        self.source_pts = self.getSourcePlane()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # Get random slice and label
        img_np, label_np = self.getRandSlice(self.volume_interp_func, self.source_pts)
        img = torch.from_numpy(img_np).float()
        img = img.view(1, self.slice_sz, self.slice_sz)
        label = torch.from_numpy(label_np).float()
        sample = {'img': img, 'label': label, 'index': idx}
        return sample

    def getVolumeInterpFunc(self, dirName='./test'):
        # Read volume data from /test folder
        dirname = dirName
        files = os.listdir(dirname)
        ds_list = [pydicom.filereader.dcmread(os.path.join(dirname, filename)) for filename in files]
        rawx, rawy = ds_list[0].pixel_array.shape
        rawz = len(ds_list)

        def takeSpaceLocation(elem):
            return elem.SliceLocation

        ds_list.sort(key=takeSpaceLocation)
        data_volume = np.zeros([rawx, rawy, rawz])
        for i in range(rawz):
            data_volume[:, :, i] = ds_list[i].pixel_array

        # Spacing grids
        x = np.linspace(0, float(ds_list[0].PixelSpacing[0]) * rawx, rawx, endpoint=False)
        y = np.linspace(0, float(ds_list[0].PixelSpacing[1]) * rawy, rawy, endpoint=False)
        z = np.linspace(float(ds_list[0].SliceLocation), float(ds_list[0].SliceLocation) + \
                        float(ds_list[0].SpacingBetweenSlices) * rawz, rawz, endpoint=False)

        # Interpolation
        volume_interp_func = RegularGridInterpolator((x, y, z), data_volume, bounds_error=False, fill_value=0)

        return volume_interp_func

    def getSourcePlane(self):
        # Create source plane
        slice_sz = self.slice_sz
        source_pts = np.zeros([4, slice_sz * slice_sz])
        for i in range(slice_sz):
            for j in range(slice_sz):
                source_pts[0, i * slice_sz + j] = i - slice_sz / 2
                source_pts[1, i * slice_sz + j] = j - slice_sz / 2
                source_pts[3, i * slice_sz + j] = 1
        return source_pts

    def getRandSlice(self, volume_interp_func, source_pts):
        # Generate random slice
        slice_sz = self.slice_sz
        f = randTrans4x4(debug=False)
        trans_pts = np.matmul(f, source_pts)
        trans_pts[0:2, :] = trans_pts[0:2, :] + slice_sz / 2
        trans_pts = np.transpose(trans_pts)
        interp_vals = volume_interp_func(trans_pts[:, 0:3])
        random_slice = np.reshape(interp_vals, (slice_sz, slice_sz))

        source_anchor_pts = np.transpose(np.array([[0, 0, 0, 1], [slice_sz / 2, slice_sz, 0, 1], [slice_sz, 0, 0, 1]]))
        trans_anchor_pts = np.matmul(f, source_anchor_pts)
        label = trans_anchor_pts[0:3, :].flatten('F')

        return random_slice, label

class SliceDataSet(Dataset):
    """slice data set."""

    def __init__(self, data_dir='../data_train/', slice_sz = 400):
        self.data_dir = data_dir
        self.labels = self.parseFiles(data_dir)
        self.slice_sz = slice_sz

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # get item based on index
        img_path = os.path.join(self.data_dir, 'img_(%d).png' % idx)
        imgPIL = Image.open(img_path)
        imgPIL.load()
        imgdata = np.asarray(imgPIL, dtype="uint8")
        img = torch.from_numpy(imgdata).float()
        img = img.view(1, self.slice_sz, self.slice_sz)

        label = torch.FloatTensor(self.labels[idx][-9:])
        sample = {'img': img, 'label': label, 'index': idx}

        return sample

    def parseFiles(self, dirName='../data_train/'):
        label_path = os.path.join(dirName, 'label.csv')
        with open(label_path) as csvfile:
            data = csv.reader(csvfile, delimiter=",")
            labels = []
            rownum = 0
            for row in data:
                labels.append(row)
                labels[rownum] = [float(i) for i in labels[rownum]]
                rownum += 1
        return labels


# Define deep neural network
class HashingNet(nn.Module):

    def __init__(self):
        super(HashingNet, self).__init__()
        self.nn1 = nn.Sequential(
            nn.Conv2d(1, 96, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, stride=4),
            nn.LocalResponseNorm(2),

            nn.Conv2d(96, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, stride=4),
            nn.LocalResponseNorm(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(5, stride=5),
            nn.LocalResponseNorm(2),
        )

        self.nn2 = nn.Sequential(
            nn.Linear(6400, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 9),
        )

    def forward(self, x):
        temp = self.nn1(x)
        temp = temp.view(temp.size(0), -1)
        output = self.nn2(temp)
        return output

# Define deep neural network
class HashingNetBinary(nn.Module):

    def __init__(self):
        super(HashingNetBinary, self).__init__()
        self.nn1 = nn.Sequential(
            nn.Conv2d(1, 96, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, stride=4),
            nn.LocalResponseNorm(2),

            nn.Conv2d(96, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, stride=4),
            nn.LocalResponseNorm(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(5, stride=5),
            nn.LocalResponseNorm(2),
        )

        self.nn2 = nn.Sequential(
            nn.Linear(6400, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 32),
        )

    def forward(self, x):
        temp = self.nn1(x)
        temp = temp.view(temp.size(0), -1)
        output = self.nn2(temp)
        return output

if __name__ == "__main__":
    weights_dir = './params.pth.tar'

    # Training process setup
    # slice_train = SliceDataSet(data_dir='../data_train/')
    slice_train = SliceDataSetUnlimited(data_dir='../test')
    train_loader = DataLoader(slice_train, batch_size=configs['batch_train'], shuffle=False, num_workers=configs['num_workers'])

    # Training the net
    net = HashingNet().cuda()
    optimizer = optim.Adam(net.parameters(), lr = configs['learning_rate'])
    loss_fn = nn.MSELoss()
    total_epoch = configs['epochs']
    counter = []
    loss_history = []
    iteration = 0

    # def weights_init(m):
    #     if isinstance(m, nn.Conv2d):
    #         xavier(m.weight.data)
    #         xavier(m.bias.data)
    #
    # model.apply(weights_init)

    for epoch in range(total_epoch):
        for batch_idx, batch_sample in enumerate(train_loader):
            img = batch_sample['img']
            label = batch_sample['label']
            img, y = Variable(img).cuda(), Variable(label).cuda()
            optimizer.zero_grad()
            y_pred = net(img)
            mse_loss = loss_fn(y_pred, y)
            mse_loss.backward()
            optimizer.step()

            if batch_idx % (len(slice_train)/configs['batch_train']/5) == 0:
                print("Epoch %d, Batch %d Loss %f" % (epoch, batch_idx, mse_loss.item()))
                iteration += 20
                counter.append(iteration)
                loss_history.append(mse_loss.item())

    torch.save(net.state_dict(), weights_dir)
    total_hist = [counter, loss_history]
    with open("training_hist.txt", "wb") as fp:
        pickle.dump(total_hist, fp)


