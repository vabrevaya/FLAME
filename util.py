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