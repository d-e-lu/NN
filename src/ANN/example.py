import mnist_loader
import numpy as np
import ANN
import PygamePaint
from PIL import Image

def main():

    lsizes = np.array(([784], [50], [30], [10]))
    learning_rate = 0.4
    bias_learning_rate = 0.4
    n = ANN.ArtificialNeuralNet(lsizes, ANN.squared_error, learning_rate, bias_learning_rate)
    training_data, validation_data, testing_data = mnist_loader.load_data_wrapper()


    nc = ANN.NetworkChecker()

    if nc.check(n, training_data[0][0], training_data[0][1].T.ravel()):
        n.train(training_data)
        percentage, correct, incorrect = n.test(testing_data=testing_data, output_size=training_data[0][1].size)
        print percentage, "%"
        print "Correct [1-9]", correct
        print "Incorrect [1-9]", incorrect
        # n.save_weights(percentage, lsizes, learning_rate)

    user_input = raw_input("Please select image file to detect. Or type q to quit.\n")

    while True:
        if user_input == "q":
            break
        elif user_input == "draw":
            PygamePaint.draw()
            raw_image = Image.open("screenshot.png").convert('L')
        else:
            try:
                raw_image = Image.open(user_input).convert('L')
            except:
                print "Not a valid file. Please Try again\n"
                user_input = raw_input("Please select image file to detect. Or type q to quit.\n")
                continue

        scaled_image = raw_image.resize((28,28))
        scaled_image.save("scaled.png")
        image = np.reshape(np.asarray(scaled_image), (1,784))
        image = np.true_divide(image, 255)

        vals = n.forward(image)
        print "I see a {}".format(np.argmax(vals))
        print vals
        user_input = raw_input("Please select image file to detect. Or type q to quit.\n")


if __name__ == "__main__":
    main()
