import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pfilter import ParticleFilter, gaussian_noise, squared_error, independent_sample
from scipy.stats import norm, gamma, uniform
from matplotlib.animation import FuncAnimation

# ------------- Open data -------------
#
with open("../data_series/series5/label.csv") as csvfile:
    data = csv.reader(csvfile, delimiter=",")
    labels = []
    rownum = 0
    for row in data:
        labels.append(row)
        labels[rownum] = [float(i) for i in labels[rownum]]
        rownum += 1

with open("../data_series/series5/predict.csv") as csvfile:
    data = csv.reader(csvfile, delimiter=",")
    preds = []
    rownum = 0
    for row in data:
        preds.append(row)
        preds[rownum] = [float(i) for i in preds[rownum]]
        rownum += 1

nppreds = np.asarray(preds)
nplabel = np.asarray(labels)

# ------------- Particle filter setup -------------
#
columns = ["x", "y", "z", "dx", "dy", "dz"]
prior_fn = independent_sample(
    [
        # Location
        norm(loc=nppreds[0,0], scale=20).rvs,
        norm(loc=nppreds[0,1], scale=20).rvs,
        norm(loc=nppreds[0,2], scale=20).rvs,
        # Speed
        norm(loc=0, scale=0.5).rvs,
        norm(loc=0, scale=0.5).rvs,
        norm(loc=0, scale=0.5).rvs,
    ]
)

def ob_fn(x):
    # observation function
    y = np.zeros((x.shape[0], 3))
    for i, particle in enumerate(x):
        y[i,0] = norm(loc=particle[0], scale=2).rvs(1)
        y[i,1] = norm(loc=particle[1], scale=2).rvs(1)
        y[i,2] = norm(loc=particle[2], scale=2).rvs(1)
    return y

# very simple linear dynamics: x += dx
def velocity(x):
    dt = 1.0
    xp = (
        x
        @ np.array(
            [
                [1, 0, 0, dt, 0],
                [0, 1, 0, 0, dt],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        ).T
    )

    return xp

def process_fn(x):
    # Dynamic process function
    dt = 1.0
    xp = (
        x
        @ np.array([[1.,0,0,dt, 0, 0],
                 [0,1.,0,0, dt, 0],
                 [0,0,1.,0, 0, dt],
                 [0,0,0,1., 0, 0],
                 [0,0,0,0, 1., 0],
                 [0,0,0,0, 0, 1.]]).T
    )
    return xp

def rbf_error(x, y, sigma=1):
    # RBF kernel
    d = np.sum((x - y) ** 2, axis=1)
    return np.exp(-d / (2.0 * sigma ** 2))

pf = ParticleFilter(
    prior_fn=prior_fn,
    observe_fn=ob_fn,
    n_particles=100,
    dynamics_fn=process_fn,
    noise_fn=lambda x: gaussian_noise(x, sigmas=[5, 5, 5, 0.05, 0.05,0.1]),
    weight_fn=lambda x, y: rbf_error(x, y, sigma=2),
    resample_proportion=0.2,
    column_names=columns,
)

# ------------- Running Particle filter with animation -------------
#
pf.init_filter()

class UpdateParticle(object):
    def __init__(self, ax, preds, labels):
        self.nppreds = preds
        self.nplabel = labels
        self.ax = ax

        # Set up plot parameters
        ax.set_xlim3d([0, 250])
        ax.set_ylim3d([-350, -100])
        ax.set_zlim3d([-250, 0])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')

        x = self.nplabel[:, 3]
        y = self.nplabel[:, 4]
        z = self.nplabel[:, 5]
        ax.plot(x, y, z, label='ground truth', c='g')

    def init(self):
        # Set up plot parameters
        ax.set_xlim3d([0, 250])
        ax.set_ylim3d([-350, -100])
        ax.set_zlim3d([-250, 0])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')

        x = self.nplabel[:, 3]
        y = self.nplabel[:, 4]
        z = self.nplabel[:, 5]
        ax.plot(x, y, z, label='ground truth', c='g')

        return

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        self.ax.clear()
        self.init()

        pf.update(self.nppreds[i + 1, 0:3])

        xa = pf.particles[:,0]
        ya = pf.particles[:,1]
        za = pf.particles[:,2]
        self.ax.scatter(xa,ya,za)

        x_hat, y_hat, z_hat, dx_hat, dy_hat, dz_hat = pf.mean_state
        self.ax.scatter(x_hat,y_hat,z_hat, s=80, c='r', marker='^')

        return

fig = plt.figure()
ax = fig.gca(projection='3d')
ud = UpdateParticle(ax, nppreds, nplabel)
anim = FuncAnimation(fig, ud, frames=np.arange(nppreds.shape[0]-1), init_func=ud.init,
                     interval=200, blit=False)
plt.show()


# # ------------- Running Particle filter -------------
# #
# pf.init_filter()
# pred_filt_rec = np.zeros([nppreds.shape[0]-1, 3])
# for i in range(nppreds.shape[0]-1):
#     pf.update(nppreds[i+1,0:3])
#     x_hat, y_hat, z_hat, dx_hat, dy_hat, dz_hat = pf.mean_state
#     pred_filt_rec[i,0] = x_hat
#     pred_filt_rec[i,1] = y_hat
#     pred_filt_rec[i,2] = z_hat
#
#
# # ------------- Numerical comparison -------------
# #
# total_diff_1 = 0
# total_diff_2 = 0
# for i in range(nplabel.shape[0]-1):
#     diff1 = np.linalg.norm(nplabel[i+1,3:6] - nppreds[i+1,0:3])
#     total_diff_1 += diff1
#     diff2 = np.linalg.norm(nplabel[i+1,3:6] - pred_filt_rec[i,0:3])
#     total_diff_2 += diff2
# print("Mean error before filt: ", total_diff_1/nplabel.shape[0])
# print("Mean error after filt: ", total_diff_2/nplabel.shape[0])
#
# # -------------  Plot the trajectories -------------
# #
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# x = nplabel[:,3]
# y = nplabel[:,4]
# z = nplabel[:,5]
# ax.plot(x, y, z, label='ground truth')
#
# x = nppreds[:,0]
# y = nppreds[:,1]
# z = nppreds[:,2]
# ax.plot(x, y, z, label='raw predicts')
#
# x = pred_filt_rec[:,0]
# y = pred_filt_rec[:,1]
# z = pred_filt_rec[:,2]
# ax.plot(x, y, z, label='filtered predicts')
#
# ax.legend()
# ax.set_xlim3d([0, 250])
# ax.set_ylim3d([-350, -100])
# ax.set_zlim3d([-250, 0])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_aspect('equal')
# # ax.view_init(elev=20., azim=-35)
#
# plt.show()