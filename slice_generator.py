
import numpy as np
import pydicom
import os
from helper import randTrans4x4
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
# from scipy.ndimage.interpolation import rotate

# Choose between datasets
data_type = 'Pancreas' # or 'Liver'

if data_type == 'Liver':
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
    slice_sz = 512
    source_pts = np.zeros([4, slice_sz*slice_sz])
    for i in range(slice_sz):
        for j in range(slice_sz):
            source_pts[0, i * slice_sz + j] = i - slice_sz / 2
            source_pts[1, i * slice_sz + j] = j - slice_sz / 2
            source_pts[3, i * slice_sz + j] = 1

    # Generate random slice
    f = randTrans4x4(debug=False)
    print(f)
    trans_pts = np.matmul(f, source_pts)
    for i in range(slice_sz):
        for j in range(slice_sz):
            trans_pts[0, i*slice_sz + j] = trans_pts[0, i*slice_sz + j] + slice_sz/2
            trans_pts[1, i*slice_sz + j] = trans_pts[1, i*slice_sz + j] + slice_sz/2
    trans_pts = np.transpose(trans_pts)
    interp_vals = volume_interp_func(trans_pts[:,0:3])
    random_slice  = np.reshape(interp_vals,(slice_sz,slice_sz))
    # print(random_slice)

    # Visualization
    plt.imshow(random_slice)
    plt.show()

elif data_type == 'Pancreas':
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
        data_volume[:,:,i] = ds_list[i].pixel_array

    # Spacing grids
    x = np.linspace(0, float(ds_list[0].PixelSpacing[0])*rawx, rawx, endpoint=False)
    y = np.linspace(0, float(ds_list[0].PixelSpacing[1])*rawy, rawy, endpoint=False)
    z = np.linspace(float(ds_list[0].ImagePositionPatient[2]), float(ds_list[0].ImagePositionPatient[2]) + \
                    float(abs(ds_list[0].ImagePositionPatient[2] - ds_list[1].ImagePositionPatient[2])) * \
                    rawz, rawz, endpoint=False)

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

    # Generate random slice
    f = randTrans4x4(debug=False)
    print(f)
    trans_pts = np.matmul(f, source_pts)
    for i in range(slice_sz):
        for j in range(slice_sz):
            trans_pts[0, i*slice_sz + j] = trans_pts[0, i*slice_sz + j] + slice_sz/2
            trans_pts[1, i*slice_sz + j] = trans_pts[1, i*slice_sz + j] + slice_sz/2
    trans_pts = np.transpose(trans_pts)
    interp_vals = volume_interp_func(trans_pts[:,0:3])
    random_slice  = np.reshape(interp_vals,(slice_sz,slice_sz))
    # print(random_slice)

    # Visualization
    plt.imshow(random_slice)
    plt.show()

else:
    print("Wrong data type specified! ")