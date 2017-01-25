from PIL import Image
import numpy as np

#Opens image
im = Image.open('peacock.png')

#prints size of array (640, 480), pixel format (RGB), and file format(PNG)
print im.size, im.mode, im.format

# convert the image to a black and white "luminance" greyscale image
im = im.convert('L')

#crop image
im2 = im.crop((475,130,575,230))
im2.save('peacock_head.png', 'PNG')

#convert to numpy array
im2array = np.asarray(im2)
#compute average intensity
average = np.mean(im2array)

im3array = im2array.copy()
for y in range(len(im3array)):
    for x in range(len(im3array[0])):
        im3array[y,x] = min(im2array[y,x] + 50, 255)

im3 = Image.fromarray(im3array)
im3.save('peacock_head_bright.png','PNG')

im4array = im2array.copy()
im4array = im4array * 0.5
im4array = im4array.astype('uint8')
im4 = Image.fromarray(im4array)
im4.save('peacock_head_dark.png','PNG')

grad = np.arange(0,256)
grad = np.tile(grad,[256,1])
im5 = Image.fromarray(grad.astype('uint8'))
im5.save('gradient.png','PNG')
