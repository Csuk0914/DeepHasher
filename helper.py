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