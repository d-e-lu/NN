import numpy as np
from PIL import Image

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
def image_to_greyscale_array(filepath):
    img = Image.open(filepath)
    grey = img.convert('L')
    return np.array(grey)

def image_to_colour_array(filepath):
    img = Image.open(filepath)
    return np.array(img)

def save_image(img, name):
    im = Image.fromarray(img)
    im.save(name)

class Complex_Layer(object):
    def __init__(self, con_size, pool_size, depth):
        self.con_size = con_size
        self.pool_size = pool_size
        self.pool_stride = pool_size
        self.depth = depth
        self.con_filter = np.random.randn(con_size,con_size)
        self.pool_matrix = np.random.randn(pool_size, pool_size)

    def convolve(self, image, factor=1, bias=0):
        self.con_filter = np.rot90(self.con_filter, 2) #rotates the kernel by 180 degrees
        new_image = np.empty_like(image)
        half_size = self.con_size//2
        for y in range(len(image[:,0])-(self.con_size-1)): #For row in image
            for x in range(len(image[0])-(self.con_size-1)): #For RGB pixels in row
                for z in range(len(image[0,0,:])): #For pixel in RGB
                    p = np.sum(np.multiply(image[y:y+self.con_size, x:x+self.con_size,z], self.con_filter)) * factor + bias
                    p = min(255, p)
                    p = max(0, p)
                    new_image[y + half_size,x + half_size,z] = p #Input in position at the middle of the kernel on new image
        return new_image

    def max_pool(self, image):
        new_image = np.zeros((image.shape[0]//self.pool_stride, image.shape[1]//self.pool_stride, image.shape[2]), np.uint8)
        for y in range(0,len(image[:,0])-(self.pool_size-1), self.pool_stride):
            for x in range(0,len(image[0])-(self.pool_size-1), self.pool_stride):
                for z in range(len(image[0,0,:])):
                    sub_image = image[y:y+self.pool_size, x:x+self.pool_size,z]
                    c = np.argmax(sub_image)
                    new_image[y//self.pool_stride,x//self.pool_stride,z] = sub_image.ravel()[c]
        return new_image

    def RelU(self, x):
        return np.maximum(0,x)

    def forward(self, image):
        convolved = self.convolve(image)
        non_linear = self.RelU(convolved)
        return self.max_pool(non_linear)

class CNN(object):
    def __init__(self, complex_layers, fully_connected_sizes):
        self.complex_layers = complex_layers
        self.fully_connected_sizes = fully_connected_sizes

    def forward(self, image):
        pass

def main():
    image = image_to_colour_array("pictures/img.jpg")
    cl = Complex_Layer(3, 2, 1)
    new_image = cl.forward(image)
    save_image(new_image, "pictures/weird.jpg")

if __name__ == "__main__":
    main()
