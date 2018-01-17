import numpy as np
from PIL import Image

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

def normalize_colour(x):
    return np.divide(x, 255)


def standalone_convolution(image, kernel, factor = 1.0, bias = 0.0):
    kernel = np.rot90(kernel, 2) #rotates the kernel by 180 degrees
    new_image = np.empty_like(image)
    k_width = len(kernel[0])
    k_height = len(kernel[:,0])
    m_width = k_width//2
    m_height = k_height//2
    for y in range(len(image[:,0])-(k_height-1)): #For row in image
        for x in range(len(image[0])-(k_width-1)): #For RGB pixels in row
            for z in range(len(image[0,0,:])): #For pixel in RGB
                p = np.sum(np.multiply(image[y:y+k_height, x:x+k_width,z], kernel)) * factor + bias
                p = min(255, p)
                p = max(0, p)
                new_image[y + m_height,x + m_width,z] = p #Input in position at the middle of the kernel on new image
    return new_image

def standalone_relu(x):
    return np.maximum(x, 0, x) # faster to do maximum in place

def standalone_relu_prime(x):
    return np.greater(x, 0).astype(int)

'''
Convolution and Pooling layers
'''


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
                    # Input in position at the middle of the kernel on new image
                    new_image[y+half_size, x+half_size, filter_index] = p
        return new_image

    def zero_padding(self, image, width=1):
        new_image = image[width:-width, width:-width, :]  # remove outer edges
        new_image.astype(np.uint8)
        new_image = np.pad(new_image, ((width, width), (width, width), (0, 0)), 'constant')  # pads image with zeros
        return new_image

    def forward(self, image):
        return self.zero_padding(self.convolve(image))


class ReluLayer(object):
    def relu(self, image):
        return np.maximum(image, 0, image)

    def forward(self, image):
        return self.relu(image)


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

    def forward(self, image):
        return self.max_pool(image)


'''
Fully Connected layers
'''


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def sigmoid_prime(z):
    return np.exp(-z)/((1 + np.exp(-z)) ** 2)


def squared_error(y, y_hat):
    j = 0.5 * sum((y - y_hat) ** 2)
    return j


class FullyConnectedLayer(object):
    def __init__(self, layer_sizes, regularization_function, learning_rate, bias_learning_rate,
                 momentum=0.0):
        self.regularization_function = regularization_function
        self.activation_function = sigmoid
        self.activation_function_prime = sigmoid_prime
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.bias_learning_rate = bias_learning_rate
        self.momentum = momentum
        self.length = len(layer_sizes)

        self.b = []  # Biases
        self.W = []  # Weights
        self.delta_W = []
        for i in range(self.length - 1):
            weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1])
            bias = np.random.randn(self.layer_sizes[i+1])
            self.W.append(weight)
            self.b.append(bias)
            self.delta_W.append(np.zeros_like(weight))
        self.a = None  # Activations of layers
        self.z = None  # Weights multiplied by input/activations
        self.y_hat = None  # Output of neural net

    def forward(self, x):
        self.a = []
        self.z = []
        self.z.append(np.add(np.dot(x, self.W[0]), self.b[0]))
        self.a.append(self.activation_function(self.z[0]))
        for weight in range(1, len(self.W)):
            self.z.append(np.add(np.dot(self.a[weight-1], self.W[weight]), self.b[weight]))
            self.a.append(self.activation_function(self.z[weight]))
        return self.a[-1]

    def loss_function_prime(self, x, y):
        self.y_hat = self.forward(x)
        dj_dw_list = []
        dj_db_list = []

        delta = np.multiply(-(y - self.y_hat), self.activation_function_prime(self.z[self.length-2]))
        dj_db_list.append(delta)
        dj_dw = np.multiply(self.a[self.length-3][np.newaxis].T, delta[np.newaxis])
        dj_dw_list.append(dj_dw)

        for i in range(self.length-3, 0, -1):
            delta = np.dot(delta, self.W[i+1].T) * self.activation_function_prime(self.z[i])
            dj_db_list.append(delta)
            dj_dw = np.multiply(self.a[i-1][np.newaxis].T, delta)
            dj_dw_list.append(dj_dw)

        delta = np.dot(delta, self.W[1].T) * self.activation_function_prime(self.z[0])
        dj_db_list.append(delta)
        x_array = np.array([x])

        dj_dw = np.multiply(x_array.T, delta)
        dj_dw_list.append(dj_dw)
        dj_dw_list.reverse()
        dj_db_list.reverse()
        return dj_dw_list, dj_db_list, delta

    def back_prop(self, x, y):
        dj_dw_list, dj_db_list, delta = self.loss_function_prime(x, y)
        if self.momentum:
            for i in range(len(dj_dw_list)):
                current_weight = self.W[i]
                self.W[i] = self.W[i] - (dj_dw_list[i] * self.learning_rate) + (self.momentum * self.delta_W[i])
                self.delta_W[i] = self.W[i] - current_weight
                self.b[i] = self.b[i] - (dj_db_list[i] * self.bias_learning_rate)
        else:
            for i in range(len(dj_dw_list)):
                self.W[i] = self.W[i] - (dj_dw_list[i] * self.learning_rate)
                self.b[i] = self.b[i] - (dj_db_list[i] * self.bias_learning_rate)
        yh = self.forward(x)
        return delta, squared_error(y, yh)

    def train(self, training_data):
        total_cost = 0
        for z in range(1):
            for i in range(len(training_data)):
                cost = self.back_prop(training_data[i][0], training_data[i][1].T.ravel())
                total_cost += cost
                if i % 1000 == 0:
                    avg = float(total_cost) / 1000
                    print ("data #", i, "cost:", avg)
                    total_cost = 0

    def test(self, testing_data, output_size):
        incorrect = np.zeros(output_size, dtype=np.int)
        correct = np.zeros(output_size, dtype=np.int)
        number_correct = 0
        total = 0
        for i in range(len(testing_data)):
            z = self.forward(testing_data[i][0])
            if np.argmax(z) == testing_data[i][1]:
                number_correct += 1
                correct[testing_data[i][1]] += 1
            else:
                incorrect[testing_data[i][1]] += 1
            total += 1
        percentage = float(number_correct) / total * 100

        return percentage, correct, incorrect


class CNN(object):
    def __init__(self, layers, fully_connected):
        self.layers = layers
        self.fully_connected = fully_connected

    def forward(self, image):
        new_image = image
        outputs = []
        for layer in self.layers:
            new_image = layer.forward(new_image)
            outputs.append(new_image)
        new_image = new_image.ravel()
        return new_image, outputs

    def back_prop(self, image, label):
        convolution_layer_estimate, outputs = self.forward(image)

        dj_dw_list = []

        delta, error = self.fully_connected.back_prop(normalize_colour(convolution_layer_estimate), label)
        delta = np.dot(delta, self.fully_connected.W[0].T) * standalone_relu_prime(convolution_layer_estimate)


        '''
        for i in range(len(self.layers)):
            if self.layers[i] is ConvolutionLayer:
                pass
            elif self.layers[i] is ReluLayer:
                pass
            elif self.layers[i] is PoolingLayer:
                pass

        return error
        '''

    def train(self, training_data):
        pass
