import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from shutil import copyfile

import numpy as np

def dict2obj(d):
    if not isinstance(d, dict):
        return d
    class C(object):
        pass
    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o

def cv2_plot_annotations(img, kpts=None, bboxes=None, point_rad=5, color=None):
    """
    TODO comment
    kpts and bboxes are lists
    """

    assert kpts is None or isinstance(kpts, list) or isinstance(kpts, np.ndarray)
    assert bboxes is None or isinstance(bboxes, list) or isinstance(bboxes, np.ndarray)
    # assert bboxes is None or len(kpts) == len(bboxes)

    _img = img.copy()
    n = len(kpts) if kpts is not None else len(bboxes)

    for i in range(n):

        if color is not None:
            curr_color = color
        else:
            curr_color = COLORS[i % len(COLORS)]

        # draw keypoints
        if kpts is not None:
            curr_kpts = kpts[i].astype(np.int32)
            
            for j in range(len(curr_kpts)):
                x,y = curr_kpts[j][0], curr_kpts[j][1]
                cv2.circle(_img, (x,y), point_rad, curr_color, 2)

        # draw bounding box
        if bboxes is not None:
            left, top, right, bottom = bboxes[i].astype(np.int32)
            pt1 = (left, top)
            pt2 = (right, bottom)

            cv2.rectangle(_img, pt1, pt2, curr_color, 2)

    return _img