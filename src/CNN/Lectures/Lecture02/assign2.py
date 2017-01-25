from PIL import Image
import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt


def boxfilter(n):
    assert n%2 == 1, "Dimension must be odd"
    constant = 1/float(n*n)
    return np.full((n,n), constant)

def gaussian(x, sigma):
    return np.exp(-(x**2) / (2*sigma**2))
def gauss1d(sigma):
    length = 6 * sigma + 1
    center = length//2
    array = np.arange(-center,center+1)
    vfunc = np.vectorize(gaussian)
    return vfunc(array,sigma)
    
def gauss2d(sigma):
    oneD = gauss1d(sigma)
    twoD = signal.convolve2d(oneD[np.newaxis], oneD.T[np.newaxis])
    return twoD
'''
#Test 1
print boxfilter(3)
print boxfilter(4) assertion error
print boxfilter(5)

#Test 2
print gauss1d(0.3)
print gauss1d(0.5)
print gauss1d(1)
print gauss1d(2)
x = gauss1d(2)
z = np.arange(0,len(x))
plt.plot(z, x)
plt.show()
'''
print gauss1d(1.0)
