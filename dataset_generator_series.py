# This file generates series images from a generated dataset instead of new sampling from CT.

import numpy as np
import pydicom
import os
import imageio
from scipy.interpolate import RegularGridInterpolator
from PIL import Image
from stl import mesh
from helper import rotm2axang
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

# Generating image series from existing dataset

# source_path='../data_train_geom/'
# save_path = "../data_series/series5/"
#
# label_path = os.path.join(source_path, 'label.csv')
# with open(label_path) as csvfile:
#     data = csv.reader(csvfile, delimiter=",")
#     labels = []
#     rownum = 0
#     for row in data:
#         labels.append(row)
#         labels[rownum] = [float(i) for i in labels[rownum]]
#         rownum += 1
# nplabel = np.asarray(labels)
#
# labels_series = []
# for idx in range(len(labels)):
#     if nplabel[idx,3]>=200 and nplabel[idx,3]<=203 and nplabel[idx,4]<=-200:
#         labels_series.append([idx,nplabel[idx,0],nplabel[idx,1],nplabel[idx,2],nplabel[idx,3],nplabel[idx,4],nplabel[idx,5]])
# print("# of images filtered out: ", len(labels_series))
# nplabel_series = np.asarray(labels_series)
# npsorted = nplabel_series[nplabel_series[:,6].argsort()]
#
#
# for i in range(npsorted.shape[0]):
#     img_path = os.path.join(source_path, 'img_(%d).png' % npsorted[i,0])
#     imgPIL = Image.open(img_path)
#     img_pth = os.path.join(save_path, 'img_(%d).png' % i)
#     imgPIL.save(img_pth)
#
# npsave = npsorted[:,1:]
# label_pth = os.path.join(save_path, 'label.csv')
# np.savetxt(label_pth, npsave, delimiter=",")



# This file generates series images from a generated dataset instead of new sampling from CT.
save_path = "../data_series/series6/"

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
source_pts = np.zeros([4, slice_sz * slice_sz])
for i in range(slice_sz):
    for j in range(slice_sz):
        source_pts[0, i * slice_sz + j] = i - slice_sz / 2
        source_pts[1, i * slice_sz + j] = j - 9
        source_pts[3, i * slice_sz + j] = 1
source_anchor_pts = np.transpose(
    np.array([[0, 0, 0, 1], [-slice_sz / 2, slice_sz - 9, 0, 1], [slice_sz / 2, slice_sz - 9, 0, 1]]))

# Read skin surface mesh for placing images and us fan-shape mask
mesh_test = mesh.Mesh.from_file('./STLRead/surface_skin_LPS_simplified.stl')
num_train_imgs = mesh_test.vectors.shape[0]
us_mask = imageio.imread('./us_mask.bmp')


# Generate training dataset using parallel package
def image_gen(stl_id, img_id, save_path, num_total):
    # From surface mesh get center and normal vector
    returnval = []
    vertice_test = mesh_test.vectors[stl_id, :, :]
    normal_test = mesh_test.normals[stl_id, :]
    face_center = np.mean(vertice_test, axis=0)

    if abs(normal_test[2]) > 0.9:
        returnval.append(np.array([0]))
        return returnval

    if face_center[0] <= 133 or face_center[0] >= 138 or face_center[1] >= -250:
        returnval.append(np.array([0]))
        return returnval

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
    # print(F)

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
    returnval.append(label)
    returnval.append(surface_slice)

    if img_id % 100 == 0:
        print("Generated %d images" % img_id)
    return returnval


stl_idx = 0
img_idx = 0
series_data = []
# labels = []
while stl_idx != num_train_imgs:
    returnval = image_gen(stl_idx, img_idx, save_path, num_train_imgs)
    if returnval[0].any():
        # labels.append(returnval[0].tolist())
        img_idx += 1
        series_data.append(returnval)
    stl_idx += 1

labels = []
for i in range(len(series_data)):
    labels.append(series_data[i][0])
nplabel = np.asarray(labels)

labels_series = []
for idx in range(len(labels)):
    labels_series.append([idx,nplabel[idx,0],nplabel[idx,1],nplabel[idx,2],nplabel[idx,3],nplabel[idx,4],nplabel[idx,5]])
print("# of images filtered out: ", len(labels_series))
nplabel_series = np.asarray(labels_series)
npsorted = nplabel_series[nplabel_series[:,6].argsort()]
npsave = npsorted[:,1:]

# Save images and labels
print("Saving images to files...")
for i in range(npsorted.shape[0]):
    imgPIL = Image.fromarray(series_data[int(npsorted[i][0])][1] , 'L')
    img_pth = os.path.join(save_path, 'img_(%d).png' % i)
    imgPIL.save(img_pth)
label_pth = os.path.join(save_path, 'label.csv')
np.savetxt(label_pth, npsave, delimiter=",")

# Visualize the selected path
fig = plt.figure()
ax = fig.gca(projection='3d')

x = npsave[:,3]
y = npsave[:,4]
z = npsave[:,5]
ax.plot(x, y, z, label='ground truth')

ax.legend()
ax.set_xlim3d([100, 350])
ax.set_ylim3d([-350, -100])
ax.set_zlim3d([-250, 0])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_aspect('equal')
# ax.view_init(elev=20., azim=-35)

plt.show()