from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
import ncc


def MakePyramid(image, minsize):
    image.resize((int(x*0.75),int(y*0.75), Image.BICUBIC))
    


pyramid = MakePyramid(image, minsize)
