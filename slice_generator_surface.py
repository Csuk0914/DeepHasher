import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
import imageio
from scipy.interpolate import RegularGridInterpolator
from stl import mesh

# Read skin surface mesh for placing images
mesh_test = mesh.Mesh.from_file('./STLRead/surface_skin_LPS_simplified.stl')
vertice_test = mesh_test.vectors[16,:,:]
normal_test = mesh_test.normals[16,:]
face_center = np.mean(vertice_test,axis=0)

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
        source_pts[1, i * slice_sz + j] = j - 9 # slide up by 9 pixels
        source_pts[3, i * slice_sz + j] = 1

# generate rotation
v_y = -normal_test
v_y = v_y/np.linalg.norm(v_y)
# get v_z use init v_x
v_x = np.array([1,0,0])
v_z = np.cross(v_x, v_y)
v_z = v_z/np.linalg.norm(v_z)
# get new v_x
v_x = np.cross(v_y, v_z)
v_x = v_x/np.linalg.norm(v_x)

F = np.zeros([4, 4])
F[0:3, 0:3] = np.transpose(np.array([v_x, v_y, v_z]))
F[0:3, 3] = face_center
F[3,3] = 1.0
print(F)

trans_pts = np.matmul(F, source_pts)
trans_pts = np.transpose(trans_pts)
interp_vals = volume_interp_func(trans_pts[:,0:3])
surface_slice  = np.reshape(interp_vals.astype('uint8'),(slice_sz,slice_sz))
us_mask = imageio.imread('./us_mask.bmp')
surface_slice = np.transpose(surface_slice)
surface_slice = surface_slice * us_mask

# Visualization
plt.imshow(surface_slice)
plt.show()