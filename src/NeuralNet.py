import numpy as np
import random
import warnings
import mnist_loader

warnings.filterwarnings("ignore")


'''
TODO
- get other regularization and activation functions working
- work on tkinter gui
'''


"""
Activation Functions
"""
def sigmoid(z):
    return 1/(1+np.exp(-z))
def sigmoidPrime(z):
    return np.exp(-z)/((1+np.exp(-z))**2)

def tanH(z):
    return np.tanh(z)
def tanHPrime(z):
    return (4*np.cosh(z)**2)/(np.cosh(2*z)+1)**2


"""
Cost Functions
"""
def squared_error(y, yHat):
    J = 0.5*sum((y-yHat)**2)
    return J
def squared_error_prime(self, x, y):
    self.yHat = self.forward(x)

    delta3 = np.multiply(-(y-self.yHat), sigmoidPrime(self.z3))
    dJdW2 = np.dot(self.a2.T, delta3)/x.shape[0] + self.Lambda*self.W2

    delta2 = np.dot(delta3, self.W2.T) * sigmoidPrime(self.z2)
    dJdW1 = np.dot(x.T, delta2)/x.shape[0] + self.Lambda*self.W1

    return dJdW1, dJdW2


class Stochastic_Descent(object):
    def __init__(self, layerSizes, regularization_function, activation_function, Lambda, momentum=0):
        self.regularization_function = regularization_function
        self.activation_function = activation_function
        self.layerSizes = layerSizes
        self.length = len(layerSizes)
        self.W = []
        self.delta_W = []

        for i in range(self.length - 1):
            into =  self.layerSizes[i]
            outof = self.layerSizes[i+1]
            weight = np.random.randn(into, outof)
            self.W.append(weight)
            self.delta_W.append(np.zeros_like(weight))

        self.W = np.asarray(self.W)
        self.delta_W = np.asarray(self.delta_W)
        self.Lambda = Lambda
        self.momentum = momentum

    def forward(self, x):
        self.a = []
        self.z = []
        self.z.append(np.dot(x, self.W[0]))
        self.a.append(self.activation_function(self.z[0]))
        for weight in range(1, len(self.W)):
            self.z.append(np.dot(self.a[weight-1], self.W[weight]))
            self.a.append(self.activation_function(self.z[weight]))
        return self.a[len(self.W)-1]

    def loss_function(self, x, y):
        self.yHat = self.forward(x)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def loss_function_prime(self, x, y):
        self.yHat = self.forward(x)
        dJdWList = []

        delta = np.multiply(-(y-self.yHat), sigmoidPrime(self.z[self.length-2]))
        dJdW = np.multiply(self.a[self.length-3][np.newaxis].T, delta[np.newaxis])
        dJdWList.append(dJdW)

        for i in range(self.length-3, 0,-1):
            delta = np.dot(delta, self.W[i+1].T) * sigmoidPrime(self.z[i])
            dJdW = np.multiply(np.asarray(self.a[i-1])[np.newaxis].T, np.asarray(delta))
            dJdWList.append(dJdW)

        delta = np.dot(delta, self.W[1].T) * sigmoidPrime(self.z[0])
        xarray = np.array([x])
        dJdW = np.multiply(xarray.T, delta)
        dJdWList.append(dJdW)
        dJdWList.reverse()
        return dJdWList

    def backProp(self, x,y):
        dJdWList = self.loss_function_prime(x,y)
        if self.momentum:
            for i in range(len(dJdWList)):
                currentWeight = self.W[i]
                self.W[i] = self.W[i] - dJdWList[i] * self.Lambda + self.momentum * self.delta_W[i]
                self.delta_W[i] = self.W[i] - currentWeight
        else:
            for i in range(len(dJdWList)):

                self.W[i] = self.W[i] - (dJdWList[i] * self.Lambda)

    def getParams(self):
        params = np.concatenate((self.W[0].ravel(), self.W[1].ravel()))
        for weight in range(2,self.length-1):
            params = np.concatenate((params, self.W[weight].ravel()))
        return params

    def setParams(self, params):
        w_start = 0
        w1_end = self.layerSizes[0] * self.layerSizes[1]
        self.W[0] = np.reshape(params[w_start:w1_end], (self.layerSizes[0], self.layerSizes[1]))

        for layer in range(1, self.length-1):
            w2_end = w1_end + self.layerSizes[layer] * self.layerSizes[layer+1]
            self.W[layer] = np.reshape(params[w1_end:w2_end], (self.layerSizes[layer], self.layerSizes[layer+1]))
            w1_end = w2_end

    def computeGradients(self, x, y):
        alldJdW = self.loss_function_prime(x,y)
        dJdWList = np.concatenate([alldJdW[0].ravel(), alldJdW[1].ravel()])
        for i in range(2,len(alldJdW)):
            dJdWList = np.concatenate([dJdWList, alldJdW[i].ravel()])
        return dJdWList

"""
Neural_Network Check
"""
class Network_Checker():
    def computeNumericalGradient(self,N,x,y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4
        for p in range(len(paramsInitial)):
            perturb[p]=e
            N.setParams(paramsInitial+perturb)
            loss2 = N.loss_function(x,y)
            N.setParams(paramsInitial-perturb)
            loss1 = N.loss_function(x,y)
            numgrad[p] = (loss2-loss1)/(2*e)
            perturb[p]=0

        N.setParams(paramsInitial)
        return numgrad

    def check(self,N,x,y):
        numgrad = self.computeNumericalGradient(N,x,y)
        grad = N.computeGradients(x,y)
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

"""
Train and Test data
"""
def train(n, training_data, training_output):
    l = 0
    for i in range(len(training_data)):
        n.backProp(training_data[i],training_output[i])
        l +=1
        if l % 100 == 0:
            print(l)
    print ("Neural Network has been trained")
def get_accuracy(n, testing_data, testing_output):
    correct = 0
    total = 0
    for i in range(len(testing_data)):
        z = np.argmax(n.forward(testing_data[i]))
        if z == testing_output[i]:
            correct += 1
        total += 0
    return correct/total


def main():
    lsizes = np.array(([784],[100],[100],[10]))

    n = Stochastic_Descent(lsizes, squared_error, sigmoid,1)
    training_data, validation_data, testing_data = mnist_loader.load_data_wrapper()

    for i in range(len(training_data)):
        if i % 1000  == 0:
            print i
        n.backProp(training_data[i][0], training_data[i][1].T.ravel())

    correct = 0
    total = 0
    for i in range(len(testing_data)):
        z = n.forward(testing_data[i][0])
        if np.argmax(z) == testing_data[i][1]:
            correct += 1
        total += 1
    print correct, total

if __name__ == "__main__":
    main()
