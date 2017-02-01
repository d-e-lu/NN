from PIL import Image
import numpy as np
from scipy import signal

def boxfilter(n):
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


def image_to_greyscale_array(filepath):
    img = Image.open(filepath)
    grey = img.convert('L')
    return np.array(grey)
def save_image(img, name):
    im = Image.fromarray(img)
    im = im.convert('RGB') #need this, get an IO error otherwise
    im.save(name)

def gaussconvolve2d(array, sigma):
    gauss_filter = gauss2d(sigma)
    return signal.convolve2d(array,gauss_filter,'same')#same makes result the same size as image



#Test 1
print (boxfilter(3))
#print (boxfilter(4)) assertion error
print (boxfilter(5))

#Test 2
print (gauss1d(0.3))
print (gauss1d(0.5))
print (gauss1d(1))
print (gauss1d(2))

#Test 3
print (gauss2d(0.5))
print (gauss2d(1))

#Test 4
img = image_to_greyscale_array("img.jpg")
convolved_image = gaussconvolve2d(img, 3.0)
save_image(convolved_image, "convolved.jpg")
