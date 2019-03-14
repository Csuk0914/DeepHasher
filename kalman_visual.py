import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import KalmanFilter

with open("../data_series/series5/label.csv") as csvfile:
    data = csv.reader(csvfile, delimiter=",")
    labels = []
    rownum = 0
    for row in data:
        labels.append(row)
        labels[rownum] = [float(i) for i in labels[rownum]]
        rownum += 1
print(len(labels))

with open("../data_series/series5/predict.csv") as csvfile:
    data = csv.reader(csvfile, delimiter=",")
    preds = []
    rownum = 0
    for row in data:
        preds.append(row)
        preds[rownum] = [float(i) for i in preds[rownum]]
        rownum += 1
print(len(preds))

nppreds = np.asarray(preds)
nplabel = np.asarray(labels)

# Filter setup
my_filter = KalmanFilter(dim_x=6, dim_z=3)
my_filter.x = np.array([[nppreds[0,0]],
                        [nppreds[0,1]],
                       [nppreds[0,2]],
                       [0.0],
                       [0.0],
                       [3]])                # initial state (location and velocity)
T_s = 1 # seconds
my_filter.F = np.array([[1.,0,0,T_s, 0, 0],
                        [0,1.,0,0, T_s, 0],
                        [0,0,1.,0, 0, T_s],
                        [0,0,0,1., 0, 0],
                        [0,0,0,0, 1., 0],
                        [0,0,0,0, 0, 1.]])    # state transition matrix

my_filter.H = np.array([[1.,0,0,0, 0, 0],
                       [0,1.,0,0, 0, 0],
                       [0,0,1.,0, 0, 0]])    # Measurement function
my_filter.P = np.array([[150.,0,0,0, 0, 0],
                        [0,150.,0,0, 0, 0],
                        [0,0,150.,0, 0, 0],
                        [0,0,0,1., 0, 0],
                        [0,0,0,0, 1., 0],
                        [0,0,0,0, 0, 1.]])    # covariance matrix
my_filter.R = np.array([[1000.,0,0],
                       [0,1000.,0],
                       [0,0,1000.]])            # measurement uncertainty/noise
my_filter.Q = np.array([[1,0,0,0, 0, 0],
                        [0,1,0,0, 0, 0],
                        [0,0,1,0, 0, 0],
                        [0,0,0,.1, 0, 0],
                        [0,0,0,0, .1, 0],
                        [0,0,0,0, 0, .1]])  # Process uncertainty/noise

# Start filtering process
idx = 1
pred_filt_rec = np.zeros([nplabel.shape[0] - 1, 6])
while True:
    my_filter.predict()
    my_filter.update(np.expand_dims(nppreds[idx, 0:3], axis=1))

    # do something with the output
    x = my_filter.x
    pred_filt_rec[idx - 1, :] = np.squeeze(x)

    idx += 1
    if idx == nplabel.shape[0]:
        break

# Numerical comparison
total_diff_1 = 0
total_diff_2 = 0
for i in range(nplabel.shape[0]-1):
    diff1 = np.linalg.norm(nplabel[i+1,3:6] - nppreds[i+1,0:3])
    total_diff_1 += diff1
    # print(diff1)
    diff2 = np.linalg.norm(nplabel[i+1,3:6] - np.squeeze(pred_filt_rec[i,0:3]))
    total_diff_2 += diff2
    # print(diff2)
    # print("--------New--------")
    # print(i)
print("Mean error before filt: ", total_diff_1/nplabel.shape[0])
print("Mean error after filt: ", total_diff_2/nplabel.shape[0])

# Plot the trajectories
fig = plt.figure()
ax = fig.gca(projection='3d')

x = nplabel[:,3]
y = nplabel[:,4]
z = nplabel[:,5]
ax.plot(x, y, z, label='ground truth')

x = nppreds[:,0]
y = nppreds[:,1]
z = nppreds[:,2]
ax.plot(x, y, z, label='raw predicts')

x = pred_filt_rec[:,0]
y = pred_filt_rec[:,1]
z = pred_filt_rec[:,2]
ax.plot(x, y, z, label='filtered predicts')

ax.legend(loc="upper left")
ax.set_xlim3d([0, 250])
ax.set_ylim3d([-350, -100])
ax.set_zlim3d([-250, 0])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_aspect('equal')
# ax.view_init(elev=20., azim=-35)

plt.show()