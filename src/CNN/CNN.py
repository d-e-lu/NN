import numpy as np
import mnist_loader
import pygame
from PIL import Image


'''
Convolutional neural network
1. Convolution
2. Non - linearity (ReLU)
3. Pooling or sub sampling
4. Classification
'''

display_width = 640
display_height = 480
middle = display_width/2
center = display_height/2
black = (0,0,0)
white = (255,255,255)


class CNN(object):
    def __init__(self):
        x = 1

    def relu(self,z):
        return np.max(0,z)

def main():
    img_height = 28
    img_width = 28

    block_height = display_height/img_height
    block_width = display_width/img_width

    training_data, validation_data, testing_data = mnist_loader.load_data_wrapper()
    array = training_data[20][0]
    num = training_data[20][1]
    print np.argmax(num)

    pygame.init()
    game_display = pygame.display.set_mode((display_width, display_height))

    pygame.display.set_caption("CNN")
    clock = pygame.time.Clock()

    crashed = False
    while not crashed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
        game_display.fill(black)
        offset = (display_width% img_width)/2
        w = offset
        h = 0

        for i in range(len(array)):
            c = array[i] * 255.0
            pygame.draw.rect(game_display, (c,c,c), (w,h,block_width, block_height),0)
            w += block_width
            if i % 28==0:
                w = offset
                h+=block_height
        pygame.display.update()
        clock.tick(30)


    pygame.quit()
    quit()




if __name__ == '__main__':
    main()
