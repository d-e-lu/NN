import numpy as np
from PIL import Image

'''
Convolution
- Uses a kernel to apply a filter over an image

Edge Cases:
- Wrap around image
- Ignore pixels
- Duplicate edge pixels

Important Notes:
Three steps to a complex layer
1. Convolution operation over image
2. Non - linearity (ReLU)
3. Pooling or sub sampling
4. Classification

Feature Map - output matrix/matrices of the
        convolution operation

Depth - number of filters (kernels) used which
        results in the number of output images

Stride - number of pixels by which the filter
        slides over

Zero padding - pad the image with zeros around
        the border which allows us to control the
        size of the feature maps(also called wide
        convolution)

'''
filters = {"edge_enhance" : np.array([[0, 0, 0],
                                      [-1, 1, 0],
                                      [0, 0, 0]], dtype = np.float),

            "edge_detect_1" : np.array([[0, 1, 0],
                                      [1, -4, 1],
                                      [0, 1, 0]], dtype = np.float),

            "edge_detect_2" : np.array([[-1,-1,-1],
                                        [-1,8,-1],
                                        [-1,-1,-1]], dtype = np.float),

            "emboss" : np.array([[-2, -1, 0],
                                 [-1, 1, 1],
                                 [0, 1, 2]], dtype = np.float),


            "blur" : np.array([[1,1,1],
                               [1,1,1],
                               [1,1,1]], dtype = np.float),

            "gaussian_blur" : np.array([[1,2,1],
                                        [2,4,2],
                                        [1,2,1]],dtype = np.float),

            "sharpen" : np.array([[0,-1,0],
                                  [-1,5,-1],
                                  [0,-1,0]], dtype = np.float)

            }

def ReLU(x):
    return np.max(0,x)

def image_to_greyscale_array(filepath):
    img = Image.open(filepath)
    grey = img.convert('L')
    return np.array(grey)

def image_to_colour_array(filepath):
    img = Image.open(filepath)
    return np.array(img)

def convolution(image, kernel, factor = 1.0, bias = 0.0):
    new_image = np.empty_like(image)
    k_width = len(kernel[0])
    k_height = len(kernel[:,0])
    m_width = k_width/2
    m_height = k_height/2
    for y in range(len(image[:,0])-(k_height-1)):
        for x in range(0,len(image[0])-(k_width-1)):
            for z in range(len(image[0,0,:])):
                c = np.sum(np.multiply(image[y:y+k_height, x:x+k_width,z], kernel))
                c = min(255, c)
                c = max(0, c)
                new_image[y + m_height,x + m_width,z] = (c * factor + bias)
    return new_image


def max_pooling(image, kernel, stride):
    #new_image = np.zeros((image.shape[0], image.shape[1], image.shape[2]), np.int)
    new_image = np.zeros_like(image)
    k_width = len(kernel[0])
    k_height = len(kernel[:,0])
    for y in range(0,len(image[:,0])-(k_height-1), stride):
        for x in range(0,len(image[0])-(k_width-1), stride):
            for z in range(len(image[0,0,:])):
                sub_image = image[y:y+k_height, x:x+k_width,z]
                c = np.argmax(sub_image)
                new_image[y/stride,x/stride,z] = sub_image.ravel()[c]
    return new_image

def normalize(array):
    total = np.sum(array)
    return array/total


def save_image(img, name):
    im = Image.fromarray(img)
    im.save(name)


def main():
    image = image_to_colour_array("img.jpg")
    kernel_pool = np.array([[0,0],
                            [0,0]])
    pooled_array = max_pooling(image, kernel_pool, 2)

    filtered_array = convolution(image, filters["edge_detect_2"])
    save_image(filtered_array, "img-convolved.jpg")
    save_image(pooled_array, "img-pooled.jpg")


if __name__ == "__main__":
    main()
