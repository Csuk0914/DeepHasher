
import numpy as np
import pydicom
import os
import multiprocessing
from helper import randTrans4x4
from scipy.interpolate import RegularGridInterpolator
from PIL import Image
from joblib import Parallel, delayed

# Set image files
num_cores = multiprocessing.cpu_count()
num_train_imgs = 500000
num_test_imgs = 10000

# Set image saving path
# for marcc
data_train_path = "../data/bjiang8/data_train/"
data_test_path = "../data/bjiang8/data_test/"
# for camp pc
# data_train_path = "X:/Baichuan_Files/data/data_train/"
# data_test_path = "X:/Baichuan_Files/data/data_test/"
# for local pc
# data_train_path = "../data_train2/"
# data_test_path = "../data_test2/"

# Read volume data from /test folder
dirname = '../test'
files = os.listdir(dirname)
ds_list = [pydicom.filereader.dcmread(os.path.join(dirname, filename)) for filename in files]
rawx, rawy = ds_list[0].pixel_array.shape
rawz = len(ds_list)
def takeSpaceLocation(elem):
    return elem.SliceLocation
ds_list.sort(key=takeSpaceLocation)
data_volume = np.zeros([rawx, rawy, rawz])
for i in range(rawz):
    data_volume[:,:,i] = ds_list[i].pixel_array

# Spacing grids
x = np.linspace(0, float(ds_list[0].PixelSpacing[0])*rawx, rawx, endpoint=False)
y = np.linspace(0, float(ds_list[0].PixelSpacing[1])*rawy, rawy, endpoint=False)
z = np.linspace(float(ds_list[0].SliceLocation), float(ds_list[0].SliceLocation) + \
                float(ds_list[0].SpacingBetweenSlices)*rawz, rawz, endpoint=False)

# Interpolation
volume_interp_func = RegularGridInterpolator((x, y, z), data_volume, bounds_error=False, fill_value=0)

# Create source plane
slice_sz = 400
source_pts = np.zeros([4, slice_sz*slice_sz])
for i in range(slice_sz):
    for j in range(slice_sz):
        source_pts[0, i * slice_sz + j] = i - slice_sz / 2
        source_pts[1, i * slice_sz + j] = j - slice_sz / 2
        source_pts[3, i * slice_sz + j] = 1
source_anchor_pts = np.transpose(np.array([[0, 0, 0, 1], [slice_sz / 2, slice_sz, 0, 1], [slice_sz, 0, 0, 1]]))

# Generate training dataset using parallel package
def image_gen(img_id):
    # Transform images
    f = randTrans4x4(debug=False)
    trans_pts = np.matmul(f, source_pts)
    trans_pts[0, :] = trans_pts[0, :] + slice_sz/2
    trans_pts[1, :] = trans_pts[1, :] + slice_sz/2
    trans_pts = np.transpose(trans_pts)
    interp_vals = volume_interp_func(trans_pts[:,0:3])
    random_slice  = np.reshape(interp_vals.astype('uint8'),(slice_sz,slice_sz))
    # Process label
    trans_anchor_pts = np.matmul(f, source_anchor_pts)
    label = trans_anchor_pts[0:3, :].flatten('F')
    label_var = np.zeros(10).tolist()
    label_var[0] = img_id
    label_var[1:10] = label
    # Save image
    im = Image.fromarray(random_slice)
    img_pth = os.path.join(data_train_path, 'img_(%d).png' % img_id)
    im.save(img_pth)
    # Print out process
    if img_id % (num_train_imgs/100) == 0:
        print("Generated %d images, of total %d images" % (img_id, num_train_imgs))
    return label_var

if num_train_imgs!=0:
    label_train = Parallel(n_jobs=num_cores, max_nbytes=None)(delayed(image_gen)(i) for i in range(num_train_imgs))
    label_pth = os.path.join(data_train_path, 'label.csv')
    label_train_np = np.asarray(label_train)
    label_train_np = label_train_np[label_train_np[:,0].argsort()]
    np.savetxt(label_pth, label_train_np, delimiter=",")

if num_test_imgs!=0:
    label_test = Parallel(n_jobs=num_cores, max_nbytes=None)(delayed(image_gen)(i) for i in range(num_test_imgs))
    label_pth = os.path.join(data_test_path, 'label.csv')
    label_test_np = np.asarray(label_test)
    label_test_np = label_test_np[label_test_np[:, 0].argsort()]
    np.savetxt(label_pth, label_test_np, delimiter=",")


# # Serial programming for generating testing dataset
# for img_id in range(num_test_imgs):
#     # Transform images
#     f = randTrans4x4(debug=False)
#     trans_pts = np.matmul(f, source_pts)
#     for i in range(slice_sz):
#         for j in range(slice_sz):
#             trans_pts[0, i*slice_sz + j] = trans_pts[0, i*slice_sz + j] + slice_sz/2
#             trans_pts[1, i*slice_sz + j] = trans_pts[1, i*slice_sz + j] + slice_sz/2
#     trans_pts = np.transpose(trans_pts)
#     interp_vals = volume_interp_func(trans_pts[:,0:3])
#     random_slice  = np.reshape(interp_vals.astype('uint8'),(slice_sz,slice_sz))
#     # Process label
#     trans_anchor_pts = np.matmul(f, source_anchor_pts)
#     label = trans_anchor_pts[0:3, :].flatten('F')
#     label_test[img_id, :] = label
#     # Save image
#     im = Image.fromarray(random_slice)
#     img_pth = os.path.join(data_test_path, 'img_(%d).png' % img_id)
#     im.save(img_pth)
#     # Print out process
#     if img_id % (num_test_imgs/100) == 0:
#         print("Generated %d images, of total %d images" % (img_id, num_test_imgs))
#
# label_pth = os.path.join(data_test_path, 'label.csv')
# np.savetxt(label_pth, label_test, delimiter=",")
