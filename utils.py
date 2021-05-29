from PIL import Image
import numpy as np


def read_image():
    # creating a image object
    im = Image.open("im.jpg")
    # px = im.load()
    # print(px[4, 4])
    pixels = list(im.getdata())
    im_np = np.array(pixels)
    print(im_np.shape)
    return im_np

