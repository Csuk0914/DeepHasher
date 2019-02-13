
# Evaluate training result on test dataset
from hashingNet_geom import HashingNet, SliceDataSetGeom, configs
from helper import compute_error_geom

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    print("running in eval() main function...")

    weights_dir = "./params_surface4.pth.tar"
    net = HashingNet().cuda()
    net.load_state_dict(torch.load(weights_dir))
    net.eval()
    loss_fn = nn.MSELoss()
    slice_test = SliceDataSetGeom(data_dir='../data/bjiang8/data_test_geom')
    test_loader = DataLoader(slice_test, batch_size=configs['batch_train'], shuffle=False, num_workers=1)
    total_loss = 0.0
    total_diff_center = 0.0
    total_diff_normal = 0.0

    for batch_idx, batch_sample in enumerate(test_loader):
        img = batch_sample['img']
        label = batch_sample['label']
        img, y = img.cuda(), label.cuda()
        y_pred = net(img)
        y_pred_np = y_pred.to(torch.device("cpu")).detach().numpy()
        y_np = y.to(torch.device("cpu")).detach().numpy()

        diff_center, diff_normal = compute_error_geom(y_np, y_pred_np)
        
        total_diff_center += diff_center
        total_diff_normal += diff_normal
        # if batch_idx==0:
        #     break
        if batch_idx % (len(slice_test) / configs['batch_test'] / 10) == 0:
            print("y_pred: ", y_pred[0,:])
            print("y: ", y[0,:])
            print("Batch %d: translation error %f (mm), rotation error %f (deg). " % (batch_idx, diff_center, diff_normal))

    mean_diff_center = total_diff_center / float(len(slice_test) / configs['batch_test'])
    mean_diff_normal = total_diff_normal / float(len(slice_test) / configs['batch_test'])
    print("Average translation error %f (mm), average rotation error %f (deg).", mean_diff_center, mean_diff_normal)
