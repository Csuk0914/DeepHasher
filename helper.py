# Helper function definitinos
import numpy as np

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


def randTrans4x4(dataset='pancreas',debug=False):
    """
    Generate random 4x4 transformation
    """
    if debug:
        F = np.diag([1,1,1,1])
    else:
        if dataset == 'pancreas':
            F = np.zeros([4, 4])
            F[0:3, 0:3] = randRot3()
            F[2, 3] = np.random.rand(1) * (-239.0)
            F[3, 3] = 1.0
        elif dataset == 'liver':
            F = np.zeros([4, 4])
            F[0:3, 0:3] = randRot3()
            F[2, 3] = np.random.rand(1) * 254 - 87.76
            F[3, 3] = 1.0

    return F


def compute_error(gt, re):
    # gt for ground truth, re for prediction result
    gtba = gt[:, 0:3] - gt[:, 3:6]
    gtbc = gt[:, 6:9] - gt[:, 3:6]
    gtnormal = np.cross(gtba, gtbc)
    gtnormal = gtnormal / ((np.linalg.norm(gtnormal, axis=-1))[:, None])
    gtcenter = (gt[:, 0:3] + gt[:, 3:6] + gt[:, 6:9]) / ((np.ones([gt.shape[0]]) * 3)[:, None])

    reba = re[:, 0:3] - re[:, 3:6]
    rebc = re[:, 6:9] - re[:, 3:6]
    renormal = np.cross(reba, rebc)
    renormal = renormal / ((np.linalg.norm(renormal, axis=-1))[:, None])
    recenter = (re[:, 0:3] + re[:, 3:6] + re[:, 6:9]) / ((np.ones([re.shape[0]]) * 3)[:, None])

    dotnormal = [np.degrees(np.arccos(np.dot(gtnormal[i, :], renormal[i, :]))) for i in range(gtnormal.shape[0])]
    diff_normal_avg = np.array(dotnormal).mean()

    diff_center = gtcenter - recenter
    diff_center = np.linalg.norm(diff_center, axis=-1)
    diff_center_avg = diff_center.mean()

    return diff_center_avg, diff_normal_avg