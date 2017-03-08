import mnist_loader
import numpy as np
import ANN

def main():
    lsizes = np.array(([784], [30], [10]))
    learning_rate = 0.4
    bias_learning_rate = 0.4
    n = ANN.ArtificialNeuralNet(lsizes, ANN.squared_error, learning_rate, bias_learning_rate, read_weights_from_file=False)
    training_data, validation_data, testing_data = mnist_loader.load_data_wrapper()

    n.train(training_data)
    percentage, correct, incorrect = n.test(testing_data=testing_data, output_size=training_data[0][1].size)

    print percentage, "%"
    print "Correct: ", correct
    print "Incorrect: ", incorrect
    # n.save_weights(percentage, lsizes, learning_rate)

if __name__ == "__main__":
    main()
