import mnist_loader
import numpy as np
import ANN

def main():
    lsizes = np.array(([784], [30], [10]))
    learning_rate = 0.4
    bias_learning_rate = 0.4
    n = ANN.ArtificialNeuralNet(lsizes, ANN.squared_error, learning_rate, bias_learning_rate)
    training_data, validation_data, testing_data = mnist_loader.load_data_wrapper()


<<<<<<< HEAD
    print percentage, "%"
    print "Correct: ", correct
    print "Incorrect: ", incorrect
    # n.save_weights(percentage, lsizes, learning_rate)
=======
    nc = ANN.NetworkChecker()
    
    if nc.check(n, training_data[0][0], training_data[0][1].T.ravel()):
        n.train(training_data)
        percentage, correct, incorrect = n.test(testing_data=testing_data, output_size=training_data[0][1].size)
        print percentage, "%"
        print "Correct [1-9]", correct
        print "Incorrect [1-9]", incorrect
        # n.save_weights(percentage, lsizes, learning_rate)
>>>>>>> 76b99b64e816dd43dad61e2ba679bdf0af3d395c

if __name__ == "__main__":
    main()
