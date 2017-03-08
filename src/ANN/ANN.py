import numpy as np
import codecs
import json
import os
import warnings

warnings.filterwarnings("ignore")


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def sigmoid_prime(z):
    return np.exp(-z)/((1 + np.exp(-z)) ** 2)


def squared_error(y, y_hat):
    j = 0.5 * sum((y - y_hat) ** 2)
    return j


class ArtificialNeuralNet(object):
    def __init__(self, layer_sizes, regularization_function, learning_rate, bias_learning_rate,
                 momentum=0.0, read_weights_from_file=False):
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
        if not read_weights_from_file:
            for i in range(self.length - 1):
                weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1])
                bias = np.random.randn(self.layer_sizes[i+1])
                self.W.append(weight)
                self.b.append(bias)
                self.delta_W.append(np.zeros_like(weight))
        else:
            for file_path in sorted(os.listdir("Optimal_Weights/Weights/")):
                json_file = codecs.open("Optimal_Weights/Weights/" + file_path, 'r', encoding='utf-8').read()
                json_weights = json.loads(json_file)
                weights = np.array(json_weights)
                self.W.append(weights)

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
        dJdWList = []
        dJdBList = []

        delta = np.multiply(-(y - self.y_hat), self.activation_function_prime(self.z[self.length - 2]))
        dJdBList.append(delta)
        dJdW = np.multiply(self.a[self.length-3][np.newaxis].T, delta[np.newaxis])
        dJdWList.append(dJdW)

        for i in range(self.length-3, 0, -1):
            delta = np.dot(delta, self.W[i+1].T) * self.activation_function_prime(self.z[i])
            dJdBList.append(delta)
            dJdW = np.multiply(self.a[i-1][np.newaxis].T, delta)
            dJdWList.append(dJdW)

        delta = np.dot(delta, self.W[1].T) * self.activation_function_prime(self.z[0])
        dJdBList.append(delta)
        xarray = np.array([x])
        dJdW = np.multiply(xarray.T, delta)
        dJdWList.append(dJdW)

        dJdWList.reverse()
        dJdBList.reverse()
        return dJdWList, dJdBList

    def back_prop(self, x, y):
        dJdWList, dJdBList = self.loss_function_prime(x,y)
        if self.momentum:
            for i in range(len(dJdWList)):
                current_weight = self.W[i]
                self.W[i] = self.W[i] - (dJdWList[i] * self.learning_rate) + (self.momentum * self.delta_W[i])
                self.delta_W[i] = self.W[i] - current_weight
                self.b[i] = self.b[i] - (dJdBList[i] * self.bias_learning_rate)
        else:
            for i in range(len(dJdWList)):
                self.W[i] = self.W[i] - (dJdWList[i] * self.learning_rate)
                self.b[i] = self.b[i] - (dJdBList[i] * self.bias_learning_rate)
        yh = self.forward(x)
        return squared_error(y, yh)

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

    def get_params(self):
        params = np.concatenate((self.W[0].ravel(), self.W[1].ravel()))
        for weight in range(2,self.length-1):
            params = np.concatenate((params, self.W[weight].ravel()))
        return params

    def set_params(self, params):
        w_start = 0
        w1_end = self.layer_sizes[0] * self.layer_sizes[1]
        self.W[0] = np.reshape(params[w_start:w1_end], (self.layer_sizes[0], self.layer_sizes[1]))
        for layer in range(1, self.length-1):
            w2_end = w1_end + self.layer_sizes[layer] * self.layer_sizes[layer+1]
            self.W[layer] = np.reshape(params[w1_end:w2_end], (self.layer_sizes[layer], self.layer_sizes[layer+1]))
            w1_end = w2_end

    def compute_gradients(self, x, y):
        alldJdW = self.loss_function_prime(x, y)
        dJdWList = np.concatenate([alldJdW[0].ravel(), alldJdW[1].ravel()])
        for i in range(2, len(alldJdW)):
            dJdWList = np.concatenate([dJdWList, alldJdW[i].ravel()])
        return dJdWList

    def save_weights(self, percentage, lsizes, learning_rate):
        #TODO
        if os.path.isfile("Optimal_Weights/"):
            pass
        else:
            filepath = "Optimal_Weights/Weights/W"
            for i in range(len(self.W)):
                weight_list = self.W[i].tolist()
                json.dump(weight_list, codecs.open(filepath + str(i) + ".json", 'w', encoding='utf8'), sort_keys=True, indent = 4)



"""
Neural_Network Check
"""


class NetworkChecker(object):
    def __init__(self):
        pass

    def compute_numerical_gradient(self, nn, x, y):
        params_initial = nn.get_params()
        numgrad = np.zeros(params_initial.shape)
        perturb = np.zeros(params_initial.shape)
        e = 1e-4
        for p in range(len(params_initial)):
            perturb[p] = e
            nn.set_params(params_initial + perturb)
            loss2 = nn.loss_function(x, y)
            nn.set_params(params_initial - perturb)
            loss1 = nn.loss_function(x, y)
            numgrad[p] = (loss2 - loss1)/(2 * e)
            perturb[p] = 0

        nn.set_params(params_initial)
        return numgrad

    def check(self, nn, x, y):
        numgrad = self.compute_numerical_gradient(nn, x, y)
        grad = nn.compute_gradients(x, y)
        if len(grad) != len(numgrad):
            print("Error! grad and numgrad have different sizes.")
            return False
        for i in range(len(grad)):
            difference = grad[i]-numgrad[i]
            if difference > 0.0001 or difference < -0.0001:
                print ("Error! gradient and numerical gradient are different.")
                return False
        print("Passed tests.")
        return True


"""
Normalization of Data Functions
"""


def norm_max(x, max_val=None):
    if max_val:
        return x/max_val
    else:
        return x/np.amax(x, axis=0)


def norm_percent(x):
    return x/100


def norm_standard(x):
    return (x - np.amin(x, axis=0))/(np.amax(x, axis=0) - np.amin(x, axis=0))


def shuffle_x_and_y(x, y):
    p = np.random.permutation(len(x))
    return x[p], y[p]

