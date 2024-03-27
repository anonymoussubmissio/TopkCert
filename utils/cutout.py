###############################################
# from https://github.com/uoguelph-mlrg/Cutout
###############################################

import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            # y1 = np.clip(y - self.length // 2, 0, h)
            # y2 = np.clip(y + self.length // 2, 0, h)
            # x1 = np.clip(x - self.length // 2, 0, w)
            # x2 = np.clip(x + self.length // 2, 0, w)
            y1 = np.clip(y - self.length // 2, -h*h*h,h*h*h)
            y2 = np.clip(y + self.length // 2,-h*h*h,h*h*h)
            x1 = np.clip(x - self.length // 2,-h*h*h,h*h*h)
            x2 = np.clip(x + self.length // 2,-h*h*h,h*h*h)
            # y1=fix(y1,h,0)
            # y2=fix(y2,h,0)
            # x1=fix(x1,w,0)
            # x2=fix(x2,w,0)

            mask[y1: y2, x1: x2] = 0.
            print("test")

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def fix(num,upper,lower):
    if num>=upper:
        num=(num-upper)+lower
    if num<lower:
        num=upper-(num-lower)
