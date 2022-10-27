# inspired by Toyota PackSfM repo

import cv2
import torch
import torch.nn.functional as funct
from functools import lru_cache
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def load_image(path):
    """
    Read an image using PIL

    Parameters
    ----------
    path : str
        Path to the image

    Returns
    -------
    image : PIL.Image
        Loaded image
    """
    return Image.open(path)

    
if __name__ == "__main__":
    img = load_image('data\scene0000_00\stream\depth/000000.png')
    print(img)
    pixel_map = img.load()

    for i in range(img.size[0]):    # for every col:
        for j in range(img.size[1]):    # For every row
            print(pixel_map[i,j]) # set the colour accordingly
    img.show()