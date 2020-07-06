import numpy as np
from scipy.ndimage import gaussian_filter1d

"""
Smooth the position of root joint and rotation
the position data is a numpy array with shape [frame, 3]
the rotation data is a numpy array with shape [frame, (J-1)*4]
So, the smoothing should go along with the frame axis  
"""


def gaussian_smooth(rot: np.ndarray, pos: np.ndarray):
    smooth_rot = gaussian_filter1d(rot, sigma=3, axis=0)
    smooth_pos = gaussian_filter1d(pos, sigma=3, axis=0)
    return smooth_rot, smooth_pos



