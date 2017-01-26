import numpy as np
import random
import warnings
import mnist_loader

warnings.filterwarnings("ignore")

#Neural Network using backpropogation and stochastic gradient descent


def sigmoid(z):
    return 1/(1+np.exp(-z))
def sigmoid_prime(z):
    return np.exp(-z)/((1+np.exp(-z))**2)



def squared_error(y, yHat):
    J = 0.5*sum((y-yHat)**2)
    return J

class ANN(object):
    def __init__(self, layer_sizes, regularization_function, learning_rate, momentum=0):    
        self.regularization_function = regularization_function
        self.activation_function = sigmoid
        self.activation_function_prime = sigmoid_prime
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.length = len(layer_sizes)

        #self.b = []

        self.W = []
        self.delta_W = []

        for i in range(self.length - 1):
            weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1])
            self.W.append(weight)
            self.delta_W.append(np.zeros_like(weight))

        self.W = np.asarray(self.W)
        #self.b = np.random.randn(self.W.shape)
        self.delta_W = np.asarray(self.delta_W)


    def forward(self, x):
        self.a = []
        self.z = []
        self.z.append(np.dot(x, self.W[0]))
        self.a.append(self.activation_function(self.z[0]))
        for weight in range(1, len(self.W)):
            self.z.append(np.dot(self.a[weight-1], self.W[weight]))
            self.a.append(self.activation_function(self.z[weight]))
        return self.a[len(self.W)-1]

    def loss_function_prime(self, x, y):
        self.yHat = self.forward(x)
        dJdWList = []
        delta = np.multiply(-(y-self.yHat), self.activation_function_prime(self.z[self.length-2]))
        dJdW = np.multiply(self.a[self.length-3][np.newaxis].T, delta[np.newaxis])
        dJdWList.append(dJdW)
        for i in range(self.length-3, 0,-1):
            delta = np.dot(delta, self.W[i+1].T) * self.activation_function_prime(self.z[i])
            dJdW = np.multiply(np.asarray(self.a[i-1])[np.newaxis].T, np.asarray(delta))
            dJdWList.append(dJdW)
        delta = np.dot(delta, self.W[1].T) * self.activation_function_prime(self.z[0])
        xarray = np.array([x])
        dJdW = np.multiply(xarray.T, delta)
        dJdWList.append(dJdW)
        dJdWList.reverse()
        return dJdWList

    def back_prop(self, x,y):
        dJdWList = self.loss_function_prime(x,y)
        if self.momentum:
            for i in range(len(dJdWList)):
                currentWeight = self.W[i]
                self.W[i] = self.W[i] - dJdWList[i] * self.learning_rate + self.momentum * self.delta_W[i]
                self.delta_W[i] = self.W[i] - currentWeight
        else:
            for i in range(len(dJdWList)):
                self.W[i] = self.W[i] - (dJdWList[i] * self.learning_rate)
        yh = self.forward(x)
        return squared_error(y, yh)


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
        alldJdW = self.loss_function_prime(x,y)
        dJdWList = np.concatenate([alldJdW[0].ravel(), alldJdW[1].ravel()])
        for i in range(2,len(alldJdW)):
            dJdWList = np.concatenate([dJdWList, alldJdW[i].ravel()])
        return dJdWList

"""
Neural_Network Check
"""
class Network_Checker():
    def compute_numerical_gradient(self,N,x,y):
        paramsInitial = N.get_params()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4
        for p in range(len(paramsInitial)):
            perturb[p]=e
            N.set_params(paramsInitial+perturb)
            loss2 = N.loss_function(x,y)
            N.set_params(paramsInitial-perturb)
            loss1 = N.loss_function(x,y)
            numgrad[p] = (loss2-loss1)/(2*e)
            perturb[p]=0

        N.set_params(paramsInitial)
        return numgrad

    def check(self,N,x,y):
        numgrad = self.compute_numerical_gradient(N,x,y)
        grad = N.compute_gradients(x,y)
        if(len(grad) != len(numgrad)):
            print("Error! grad and numgrad have different sizes.")
            return False
        for i in range(len(grad)):
            difference = grad[i]-numgrad[i]
            if(difference >0.0001 or difference < -0.0001):
                print ("Error! gradient and numerical gradient are different.")
                return False
        print("Passed tests.")
        return True

"""
Normalization of Data Functions
"""
def norm_max(x, max=None):
    if max:
        return x/max
    else:
        return x/np.amax(x, axis = 0)
def norm_percent(x):
    return x/100
def norm_standard(x):
    return (x - np.amin(x, axis = 0))/(np.amax(x, axis = 0) - np.amin(x, axis = 0))
def shuffle_x_and_y(x, y):
    p = np.random.permutation(len(x))
    return x[p], y[p]

def main():
    lsizes = np.array(([784],[200],[100],[10]))

    n = ANN(lsizes, squared_error, 0.5)
    training_data, validation_data, testing_data = mnist_loader.load_data_wrapper()

    total_cost = 0
    for i in range(len(training_data)):
        cost = n.back_prop(training_data[i][0], training_data[i][1].T.ravel())
        total_cost += cost
        if i % 1000  == 0:
            avg = float(total_cost)/1000
            print "data #", i,"cost:", avg
            total_cost = 0

    wrong = np.zeros(10)
    correct = 0
    total = 0
    for i in range(len(testing_data)):
        z = n.forward(testing_data[i][0])
        if np.argmax(z) == testing_data[i][1]:
            correct += 1
        else:
            wrong[testing_data[i][1]] += 1
        total += 1
    print float(correct)/total * 100, "%"
    print wrong
if __name__ == "__main__":
    main()
