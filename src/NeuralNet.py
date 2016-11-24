import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")

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

def residual(y, yHat):
    return y-yHat


class Neural_Network(object):
    def __init__(self, layerSizes, regularization_function, activation_function, Lambda, momentum=0):
        #loss function
        self.regularization_function = regularization_function
        #activation function
        self.activation_function = activation_function

        #array of sizes for each layer
        self.layerSizes = layerSizes
        #number of layers
        self.length = len(layerSizes)

        #initialize random weights and the delta weight array
        self.W = [np.random.randn(y,x) for x, y in zip(layerSizes[:-1], layerSizes[1:])]
        self.delta_W = np.zeros_like(self.W)
        self.biases = [np.random.randn(y,1) for y in layerSizes[1:]]

        #learning rate
        self.Lambda = Lambda
        #momentum
        self.momentum = momentum

    def forward(self, x):
        '''
        z = dot(inputs, weights) for a 3 layer network, z will have layers z[0] and z[1]
        a = activation(z): for a 3 layer network, a will have layers a[0] and a[1]
        '''
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

        #Computes delta and dJdW for last hidden layer to output layer.
        delta = np.multiply(-(y-self.yHat), sigmoidPrime(self.z[self.length-2]))
        dJdW = np.multiply(self.a[self.length-3].T, delta)[np.newaxis].T

        dJdWList.append(dJdW)

        #Computes delta and dJdW for all hidden layers.
        for i in range(self.length-3, 0,-1):
            delta = np.dot(delta, self.W[i+1].T) * sigmoidPrime(self.z[i])
            dJdW = np.multiply(np.asarray(self.a[i-1])[np.newaxis].T, np.asarray(delta))
            dJdWList.append(dJdW)

        #Computes delta and dJdW for input layer to first hidden layer.
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


    #Gradient Checking
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

class Batch_Descent(object):
    def __init__(self):
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1

        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

        self.Lambda = 0.0001

    def forward(self, x):
        self.z2 = np.dot(x, self.W1)
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = sigmoid(self.z3)
        return yHat

    def costFunction(self, x, y):
        self.yHat = self.forward(x)
        J = 0.5*sum((y-self.yHat)**2)/x.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J

    def costFunctionPrime(self, x, y):
        self.yHat = self.forward(x)

        delta3 = np.multiply(-(y-self.yHat), sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)/x.shape[0] + self.Lambda*self.W2

        delta2 = np.dot(delta3, self.W2.T) * sigmoidPrime(self.z2)
        dJdW1 = np.dot(x.T, delta2)/x.shape[0] + self.Lambda*self.W1

        return dJdW1, dJdW2

    def getParams(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end],(self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, x, y):
        dJdW1, dJdW2 = self.costFunctionPrime(x, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

class Boltzmann_Machine(object):
    def __init__(self, x, hiddenLayerSize):
        self.W = np.random.range(x,hiddenLayerSize)




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
        """
        Checks the correctness of the Neural Network by comparing the gradient of the cost
        with the definition of the derivative.

        Args:
            N (Neural_Network): Neural_Network to check.
            x (int, int list): One input data for N to check.
            y (int, int list): One output data for N to check.

        Returns:
            True if successful. False otherwise.
        """

        numgrad = self.computeNumericalGradient(N,x,y)
        grad = N.computeGradients(x,y)
        if(len(grad) != len(numgrad)):
            print "Error! grad and numgrad have different sizes."
            return False
        for i in range(len(grad)):
            difference = grad[i]-numgrad[i]
            if(difference >0.0001 or difference < -0.0001):
                print "Error! gradient and numerical gradient are different."
                return False
        print "Passed tests."
        return True



"""
Normalization of Data Functions
"""
def norm_max(x):
    return x/np.amax(x, axis = 0)
def norm_percent(x):
    return x/100
def norm_standard(x):
    return (x - np.amin(x, axis = 0))/(np.amax(x, axis = 0) - np.amin(x, axis = 0))
def shuffle_x_and_y(x, y):
    p = np.random.permutation(len(x))
    return x[p], y[p]

def main():

    lsizes = np.array(([2,50,50,1]))

    data = np.array(([3,5],[5,1],[10,2],[6,1.5]), dtype = float)
    output = np.array(([75],[82],[93],[70]), dtype = float)

    data = norm_max(data)
    output = norm_percent(output)
    '''
    testX = np.array(([4,5.5],[4.5,1],[9,2.5],[6,2]), dtype = float)
    testY = np.array(([70],[89],[85],[75]), dtype = float)

    testX = norm_max(testX)
    testY = norm_percent(testY)
    '''

    n = Neural_Network(lsizes, squared_error, sigmoid,0.01, 0.3)

    Network_Checker().check(n,data[1],output[1])


    for z in range(1000):
        for i in range(len(data)):
            n.backProp(data[i],output[i])


    i = 0

    for example in data:
        print "Data\n", example
        print "Estimate, Actual", n.forward(example), output[i]
        print "Cost is :\n", n.loss_function(data[i], output[i])
        print"--------------------"
        i += 1


    for i in range(len(testX)):
        print n.forward(testX[i]), testY[i]



if __name__ == "__main__":
    main()
