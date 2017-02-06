import numpy as np
from PIL import Image

filters = {"edge_enhance": np.array([[0, 0, 0],
                                     [-1, 1, 0],
                                     [0, 0, 0]], dtype=np.float),

           "edge_detect_1": np.array([[0, 1, 0],
                                      [1, -4, 1],
                                      [0, 1, 0]], dtype=np.float),

           "edge_detect_2": np.array([[-1, -1, -1],
                                      [-1, 8, -1],
                                      [-1, -1, -1]], dtype=np.float),

           "emboss": np.array([[-2, -1, 0],
                               [-1, 1, 1],
                               [0, 1, 2]], dtype=np.float),

           "blur": np.array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]], dtype=np.float),

           "gaussian_blur": np.array([[1, 2, 1],
                                      [2, 4, 2],
                                      [1, 2, 1]], dtype=np.float),

           "sharpen": np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]], dtype=np.float)
           }


def image_to_greyscale_array(filepath):
    img = Image.open(filepath)
    grey = img.convert('L')
    return np.array(grey)[:, :, np.newaxis]


def image_to_colour_array(filepath):
    img = Image.open(filepath)
    return np.array(img)


def save_image(img, name):
    im = Image.fromarray(img)
    im.save(name)


def relu(x):
    return np.maximum(0, x)


class ConvolutionLayer(object):
    def __init__(self, size, input_depth, output_depth, padding):
        self.size = size
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.padding = padding
        self.filters = np.random.randn(size, size, input_depth, output_depth)

    def convolve(self, image, factor=1, bias=0):
        assert(image.shape[2] == self.input_depth)
        half_size = self.size//2
        new_image = np.zeros((image.shape[0], image.shape[1], self.output_depth), np.uint8)
        for filter_index in range(self.output_depth):
            con_filter = np.rot90(self.filters[:, :, :, filter_index], 2)  # rotates the kernel by 180 degrees
            # Computing convolution for pixel y+half_size, x+half_size at depth filter_index
            for y in range(len(image[:, 0])-(self.size-1)):  # For every row in image
                for x in range(len(image[0])-(self.size-1)):  # For every pixel * input depth in row
                    p = np.sum(np.multiply(image[y:y+self.size, x:x+self.size], con_filter)) * factor + bias
                    p = max(0, p) # TODO: REMOVE LATER
                    p = min(255, p) # TODO: REMOVE LATER
                    # Input in position at the middle of the kernel on new image
                    new_image[y+half_size, x+half_size, filter_index] = p
        return new_image

    def zero_padding(self, image, width=1):
        new_image = image[width:-width, width:-width, :]  # remove outer edges
        new_image.astype(np.uint8)
        new_image = np.pad(new_image, ((width, width), (width, width), (0, 0)), 'constant')  # pads image with zeros
        return new_image


class PoolingLayer(object):
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def max_pool(self, image):
        new_image = np.zeros((image.shape[0]//self.stride, image.shape[1]//self.stride, image.shape[2]), np.uint8)
        for y in range(0, len(image[:, 0])-(self.size-1), self.stride):
            for x in range(0, len(image[0])-(self.size-1), self.stride):
                for z in range(len(image[0, 0, :])):
                    sub_image = image[y:y+self.size, x:x+self.size, z]
                    c = np.argmax(sub_image)
                    new_image[y//self.stride, x//self.stride, z] = sub_image.ravel()[c]
        return new_image


class CNN(object):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, image):
        pass


def main():
    image = image_to_colour_array("pictures/img.jpg")
    c_layer_1 = ConvolutionLayer(size=3, input_depth=3, output_depth=1, padding=0)
    p_layer_1 = PoolingLayer(size=2, stride=2)
    image = c_layer_1.convolve(image)
    save_image(image[:, :, 0], "32x32/convolve.jpg")
    image = relu(image)
    save_image(image[:, :, 0], "32x32/relu.jpg")
    image = p_layer_1.max_pool(image)
    save_image(image[:, :, 0], "32x32/pool.jpg")


if __name__ == "__main__":
    main()
