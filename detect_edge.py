import numpy as np
import matplotlib.pyplot as plt
import sys

#https://scikit-image.org/docs/stable/api/api.html
from skimage import filters
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.util import compare_images, invert

filename = sys.argv[1]

image = imread(filename)
image = rgb2gray(image)

edge_roberts = invert(filters.roberts(image))
edge_sobel = invert(filters.sobel(image))

imsave('roberts-' + filename, img_as_ubyte(edge_roberts))
imsave('sobel-' + filename, img_as_ubyte(edge_sobel))

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True,
                         figsize=(8, 4))

axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
axes[0].set_title('Roberts Edge Detection')

axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1].set_title('Sobel Edge Detection')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()