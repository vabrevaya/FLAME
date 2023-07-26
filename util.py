import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from shutil import copyfile

import numpy as np
from skimage.transform import estimate_transform, warp

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

def crop(
        image, 
        kpt, 
        crop_size=224,
        bbox_scale=1.,
        normalize_kpt=True, 
        return_transform=False
    ):
    """
    Crop based on keypoints

    Input:
        image   :  cv2 image?
        kpt     :  numpy array?
    """

    # assert crop_size is not None and isinstance(crop_size, int)

    left = np.min(kpt[:,0])
    right = np.max(kpt[:,0]); 
    top = np.min(kpt[:,1])
    bottom = np.max(kpt[:,1])

    h, w = image.shape[:2]
    center = np.array([left + (right - left) / 2.0, top + (bottom - top) / 2.0 ])#+ old_size*0.1])
    old_size = max(right-left, bottom-top)
    
    if isinstance(bbox_scale, list):
        bbox_scale = np.random.rand() * (bbox_scale[1] - bbox_scale[0]) + bbox_scale[0]
    
    size = int(old_size*bbox_scale)
    
    # crop 
    src_pts = np.array([\
            [center[0]-size/2, center[1]-size/2], 
            [center[0] - size/2, center[1]+size/2], 
            [center[0]+size/2, center[1]-size/2]
    ])
    dst_pts = np.array([[0,0], [0, crop_size - 1], [crop_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)

    cropped_img = warp(image, tform.inverse, output_shape=(crop_size, crop_size))
    cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
    
    if normalize_kpt:
        cropped_kpt[:,:2] = cropped_kpt[:,:2]/crop_size * 2  - 1

    if return_transform:
        return cropped_img, cropped_kpt, tform
    else:
        return cropped_img, cropped_kpt

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