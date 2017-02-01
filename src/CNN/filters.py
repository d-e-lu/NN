from PIL import Image
import numpy as np

def box_filter(n):
    assert n%2 == 1, "Dimension must be odd"
    constant = 1/float(n*n)
    return np.full((n,n), constant)

def gaussian(x, sigma):
    return np.exp(-(x**2) / (2*sigma**2))

def normalize(x):
    total = np.sum(x)
    return x/total

def gauss1d(sigma):
    length = int(6*sigma)
    if(not(length & 1)):
        length += 1
    center = length//2
    array = np.arange(-center,center+1)
    vfunc = np.vectorize(gaussian)
    return normalize(vfunc(array,sigma))

def gauss2d(sigma):
    one_d_gauss = gauss1d(sigma)
    #To get a 2d gaussian filter, take the 1d gaussian and
    #convolve it with its transpose
    two_d_gauss = signal.convolve2d(one_d_gauss[np.newaxis].T, one_d_gauss[np.newaxis])
    return two_d_gauss


def spacial(array, sigma):
    length = len(array)
    for y in range(length):
        for x in range(length):
            array[y,x] = np.exp(-(x**2+y**2)/(2*sigma**2))
    return array

def intensity(i_middle, i_current, sigma):
    return np.exp(-(i_current-i_middle)**2/(2*sigma**2))


def bilateral(array, length=3, sigma=30):
    #TODO finish filter
    b_filter = np.zeros((length, length), np.uint8)
    sp = np.zeros((length,length), np.uint8)
    sp = spacial(sp,sigma)

    new_image = np.empty_like(array)
    length = len(b_filter[0])
    half_length = length//2
    for y in range(len(array[:,0])-(length-1)):
        for x in range(len(array[0])-(length-1)):
            sub_array = array[y:y+length, x:x+length]

            #new_image[y+half_height,x+half_width] =
