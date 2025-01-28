import numpy as np
import cv2
from config import *

def image_preprocess(image,target_size,gt_box=None):
    ih, iw = IMAGE_SIZE, IMAGE_SIZE
    h, w, _ = image.shape
    
    scale = min(ih/h, iw/w)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw,nh))

    image_padded = np.full((ih,iw,3), 128)

    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_padded = image_padded / 255.

    if gt_box is None:
        return image_padded

    else:
        gt_box[:, [0,2]] = gt_box[:, [0,2]] * scale + dw
        gt_box[:, [1,3]] = gt_box[:, [1,3]] * scale + dh
        return image_padded, gt_box