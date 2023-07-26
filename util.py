import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from shutil import copyfile

import numpy as np

# colors for keypoints
COLORS = [ 
        [ 255,   0,   0],  
        [ 0,     255, 0],  
        [ 0,     0,   255],  
        [ 111,   74,   0],  
        [  81,   0,     81],  
        [ 128,   64,  128],  
        [ 244,   35,  232],  
        [ 250,   170,  160],  
        [ 230,   150,  140],  
        [  70,   70,    70],  
        [ 102,   102, 156],  
        [ 190,   153, 153],  
        [ 180,   165, 180],  
        [ 150,   100,  100],  
        [ 150,   120,   90],  
        [ 153,   153, 153],  
        [ 153,   153, 153],  
        [ 250,   170,   30],  
        [ 220,   220,    0],  
        [ 107,   142,  35],  
        [ 152,   251, 152],  
        [  70,   130,  180],  
        [ 220,   20,    60],  
        [ 255,   0,      0],  
        [   0,    0,    142],  
        [   0,    0,     70],  
        [   0,    60,   100],  
        [   0,    0,     90],  
        [   0,    0,    110],  
        [   0,    80,   100],  
        [   0,    0,    230],  
        [ 119,   11,   32],  
    ]

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