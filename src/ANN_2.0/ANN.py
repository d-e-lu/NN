import numpy as np

def affine_foward(x, w, b):
    
    x_reshape = x.reshape((x.shape[0], -1))

    out = np.dot(x_reshape, w) + b

    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    x, w, b = cache

    x_reshape = x.reshape((x.shape[0], -1))
    
    db = np.sum(dout, axis=0)
    dw = np.dot(x_reshape.T, dout)
    dx_nd = np.dot(dout, w.T)
    dx = np.reshape(dx_nd, x.shape)

    return dx, dw, db

def relu_foward(x):
    out = np.maximum(x, 0)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    x = cache
    dx = np.multiply((x>0), dout)
    return dx

def affine_relu_foward(x, w, b):
    a, affine_cache = affine_foward(x, w, b)
    out, relu_cache = relu_foward(a)
    cache = (affine_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    affine_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    return affine_backward(da, affine_cache) 
     
def softmax_loss(y_pred, y):
    shifted_logits = y_pred - np.max(y_pred, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = y_pred.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dy_pred = probs.copy()
    dy_pred[np.arange(N), y] -= 1
    dy_pred /= N
    return loss, dy_pred

def shuffle(X, y):
    assert(len(X) == len(y))
    p = np.random.permutation(len(X))
    return X[p], y[p]

class ANN(object):    
    def __init__(self, hidden_dims, input_dim, num_classes, batch_size=64, learning_rate=1e-2, weight_scale=1e-3, reg=0.0):
        """
        Initializes Dense Network
        
        Inputs:
        - hidden_dims: List of sizes of each hidden dimension
        - input_dim: Integer giving dimension of input
        - num_classes: Integer giving number of classes
        - weight_scale: Scalar giving the standard deviation from random initialization
          of the weights
        - reg: L2 regularization strength

        """
        self.reg = reg
        self.learning_rate = learning_rate
        self.num_layers = len(hidden_dims) + 1
        self.batch_size = batch_size
        self.params = {}
        self.cache = {}

        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
        self.params['b1'] = np.zeros(hidden_dims[0])

        if(self.num_layers > 1):
            curr_layer = 2
            for i in range(len(hidden_dims) - 1):
                self.params['W' + str(curr_layer)] = weight_scale * np.random.randn(hidden_dims[i], hidden_dims[i+1])
                self.params['b' + str(curr_layer)] = np.zeros(hidden_dims[i+1])
                curr_layer += 1
            self.params['W' + str(curr_layer)] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
            self.params['b' + str(curr_layer)] = np.zeros(num_classes)


    def loss(self, X, y=None):
        """
        Compute loss on minibatch of data

        Inputs:
        - X: Array of input data (N, d_1, d_2, ..., d_k)
        - Y: Array of labels of shape (N,)
        
        Returns:
        if y is None:
        - scores: Array of shape (N, C), where C is the number of classes

        if y is not None:
        - loss: Scalar loss value
        - grads: list of gradients of same shape as W
       
        """

        A, self.cache['layer1'] = affine_relu_foward(X, self.params['W1'], self.params['b1'])
        if(self.num_layers > 1):
            curr_layer = 2
            for i in range(self.num_layers - 2):
                A, self.cache['layer' + str(curr_layer)] = affine_relu_foward(
                                    A, self.params['W' + str(curr_layer)], self.params['b' + str(curr_layer)])
                curr_layer += 1
            A, self.cache['layer' + str(curr_layer)] = affine_foward(
                                    A, self.params['W' + str(curr_layer)], self.params['b' + str(curr_layer)])

        scores = A
        if y is None:
            return scores

        grads = {}
        curr_layer = self.num_layers
        loss, dscores = softmax_loss(scores, y)
        dA, grads['W' + str(curr_layer)], grads['b' + str(curr_layer)] = affine_backward(dscores, self.cache['layer' + str(curr_layer)])
        grads['W' + str(curr_layer)] += self.reg * self.params['W' + str(curr_layer)]
        loss += 0.5 * self.reg * np.sum(self.params['W' + str(curr_layer)] ** 2)
        curr_layer -= 1
        while(curr_layer > 0):
            dA, grads['W' + str(curr_layer)], grads['b' + str(curr_layer)] = affine_relu_backward(dA, self.cache['layer' + str(curr_layer)])
            grads['W' + str(curr_layer)] += self.reg * self.params['W' + str(curr_layer)]
            loss += 0.5 * self.reg * np.sum(self.params['W' + str(curr_layer)] ** 2)
            curr_layer -= 1
        return loss, grads   
    
    
    def train_batch(self, batch, y):
        batch_loss, grads = self.loss(batch, y)
        
        for i in range(1, self.num_layers+1):
            self.params["W"+str(i)] -= self.learning_rate * grads["W"+str(i)]
            self.params["b"+str(i)] -= self.learning_rate * grads["b"+str(i)]

        return batch_loss


    def train_epoch(self, X, y, shuffle_data=True):
        if shuffle_data:
            X, y = shuffle(X, y)

        num_examples = X.shape[0]
        X.reshape((num_examples, - 1))
        for i in range(int(num_examples/self.batch_size)):
            start_batch = i * self.batch_size
            end_batch = (i+1) * self.batch_size
            loss = self.train_batch(X[start_batch:end_batch],y[start_batch:end_batch])
            if i % 100 == 0:
                print(loss)

    def train(self, X, y, num_epochs):
        for i in range(num_epochs):
            print("Training epoch %d" % (i+1))
            self.train_epoch(X, y)
    
    def test(self, X, y):
        preds = self.loss(X)
        
        preds = np.round(preds)
        
        preds = np.argmax(preds, axis=1)
        acc = np.count_nonzero(preds == y)/float(len(y))
        print("accuracy %f" % acc)
        return acc   

if __name__ == "__main__":
    import tensorflow as tf
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print("X train shape: {}".format(x_train.shape))
    print("y train shape: {}".format(y_test.shape))
    print("X test shape: {}".format(x_test.shape))
    print("y test shape: {}".format(y_test.shape))

    model = ANN([100], 784, 10) 
    model.train(x_train, y_train,10)
    model.test(x_test, y_test)
