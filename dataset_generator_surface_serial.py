
import numpy as np
import pydicom
import os
import imageio
from scipy.interpolate import RegularGridInterpolator
from PIL import Image
from stl import mesh
from helper import rotm2axang

# Set image files
num_train_imgs = 50000
num_test_imgs = 5000

# Set image saving path
# for marcc
# data_train_path = "../data/bjiang8/data_train/"
# data_test_path = "../data/bjiang8/data_test/"
# for camp pc
# data_train_path = "X:/Baichuan_Files/data/data_train/"
# data_test_path = "X:/Baichuan_Files/data/data_test/"
# for local pc
data_train_path = "../data_train_geom/"
data_test_path = "../data_test_geom/"

label_type = 'geom'

if label_type == 'geom':
    label_length = 6
elif label_type == 'anchors':
    label_length = 9

# Read volume data from /test folder
dirname = '../test2'
files = os.listdir(dirname)
ds_list = [pydicom.filereader.dcmread(os.path.join(dirname, filename)) for filename in files]
rawx, rawy = ds_list[0].pixel_array.shape
rawz = len(ds_list)

def takeSpaceLocation(elem):
    return elem.ImagePositionPatient[2]

ds_list.sort(key=takeSpaceLocation)
data_volume = np.zeros([rawx, rawy, rawz])
for i in range(rawz):
    data_volume[:, :, i] = ds_list[i].pixel_array
# for aligning with stl surface data and interpolator
data_volume = np.swapaxes(data_volume, 0, 1)
data_volume = np.flip(data_volume, 1)

# Spacing grids
x = np.linspace(0, float(ds_list[0].PixelSpacing[0]) * rawx, rawx, endpoint=False)
y = np.linspace(-float(ds_list[0].PixelSpacing[1]) * (rawy - 1), 0, rawy, endpoint=True)
z = np.linspace(float(ds_list[0].ImagePositionPatient[2]),
                float(ds_list[-1].ImagePositionPatient[2]), rawz, endpoint=True)

# Interpolation
volume_interp_func = RegularGridInterpolator((x, y, z), data_volume, bounds_error=False, fill_value=0)

# Create source plane
slice_sz = 256
source_pts = np.zeros([4, slice_sz*slice_sz])
for i in range(slice_sz):
    for j in range(slice_sz):
        source_pts[0, i * slice_sz + j] = i - slice_sz / 2
        source_pts[1, i * slice_sz + j] = j - 9
        source_pts[3, i * slice_sz + j] = 1
source_anchor_pts = np.transpose(np.array([[0, 0, 0, 1], [-slice_sz / 2, slice_sz-9, 0, 1], [slice_sz / 2, slice_sz-9, 0, 1]]))

# Read skin surface mesh for placing images and us fan-shape mask
mesh_test = mesh.Mesh.from_file('./STLRead/surface_skin_LPS_simplified.stl')
us_mask = imageio.imread('./us_mask.bmp')

# Generate training dataset using parallel package
def image_gen(stl_id, img_id, save_path, num_total):
    # From surface mesh get center and normal vector
    vertice_test = mesh_test.vectors[stl_id, :, :]
    normal_test = mesh_test.normals[stl_id, :]
    face_center = np.mean(vertice_test, axis=0)

    if abs(normal_test[2]) > 0.9:
        return np.array([0])

    # generate rotation
    v_y = -normal_test
    v_y = v_y / np.linalg.norm(v_y)
    # get v_z use init v_x
    v_x = np.array([1, 0, 0])
    v_z = np.cross(v_x, v_y)
    v_z = v_z / np.linalg.norm(v_z)
    # get new v_x
    v_x = np.cross(v_y, v_z)
    v_x = v_x / np.linalg.norm(v_x)
    # Generate full transformation
    F = np.zeros([4, 4])
    F[0:3, 0:3] = np.transpose(np.array([v_x, v_y, v_z]))
    F[0:3, 3] = face_center
    F[3, 3] = 1.0
    #print(F)

    trans_pts = np.matmul(F, source_pts)
    trans_pts = np.transpose(trans_pts)
    interp_vals = volume_interp_func(trans_pts[:, 0:3])
    surface_slice = np.reshape(interp_vals.astype('uint8'), (slice_sz, slice_sz))
    surface_slice = np.transpose(surface_slice)
    surface_slice = surface_slice * us_mask

    # Process label
    if label_type == 'anchors':
        trans_anchor_pts = np.matmul(F, source_anchor_pts)
        label = trans_anchor_pts[0:3, :].flatten('F')
    elif label_type == 'geom':
        label = np.zeros(6)
        label[0:3] = rotm2axang(F[0:3, 0:3])
        label[3:6] = F[0:3, 3]
    # Save image
    im = Image.fromarray(surface_slice)
    img_pth = os.path.join(save_path, 'img_(%d).png' % img_id)
    im.save(img_pth)
    # Print out process
    if img_id % (num_total/100) == 0:
        print("Generated %d images, of total %d images" % (img_id, num_total))
    return label

stl_idx = 0
img_idx = 0

if num_train_imgs!=0:
    label_train_all = np.zeros([num_train_imgs, label_length])
    while img_idx != num_train_imgs:
        label_train = image_gen(stl_idx, img_idx, data_train_path, num_train_imgs)
        if label_train.any():
            label_train_all[img_idx, :] = label_train
            img_idx += 1
        stl_idx += 1

    label_pth = os.path.join(data_train_path, 'label.csv')
    np.savetxt(label_pth, label_train_all, delimiter=",")

if num_test_imgs!=0:
    label_test_all = np.zeros([num_test_imgs, label_length])
    while img_idx != num_train_imgs + num_test_imgs:
        label_test = image_gen(stl_idx, img_idx-num_train_imgs, data_test_path, num_test_imgs)
        if label_test.any():
            label_test_all[img_idx-num_train_imgs, :] = label_test
            img_idx += 1
        stl_idx += 1

    label_pth = os.path.join(data_test_path, 'label.csv')
    np.savetxt(label_pth, label_test_all, delimiter=",")

