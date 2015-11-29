from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage.morphology import watershed, disk
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
import numpy as np
from PIL import Image
from skimage import color
from skimage.transform import rescale, resize
from skimage.filters import gaussian_filter
import time
from skimage.transform import rescale, resize

def watershedFunc2(imagePath):
    start = time.time()
    #image = np.asarray(Image.open('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3517.JPG'))
    image = np.asarray(Image.open(imagePath))
    image = color.rgb2gray(image)
    image = rescale(image,0.25)
    denoised = gaussian_filter(image, 2)
    # denoise image
    #denoised = rank.median(image, disk(2))

    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, disk(10)) < 10
    markers = ndi.label(markers)[0]

    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, disk(2))

    # process the watershed
    labels = watershed(gradient, markers)
    end = time.time()
    print( end - start)
    return labels
# display results
def superPix(imagePath,labels):
    start = time.time()
    image = np.asarray(Image.open(imagePath))
    image = rescale(image,0.25)
    flatImage = image.reshape(image.shape[0]*image.shape[1],image.shape[2])
    print(labels.shape)
    flatLabels = labels.reshape(labels.shape[0]*labels.shape[1])
    i = 0
    while i < (np.max(flatLabels)+1):
      
        indices = (flatLabels==i)
        indices = np.dstack([indices,indices,indices])
        superPixel = indices*flatImage
        superPixel = superPixel.reshape(image.shape[0],image.shape[1],image.shape[2])
        i += 1
    end = time.time()
    print( end - start)
    return superPixel
      
'''
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
axes = axes.ravel()
ax0, ax1, ax2, ax3 = axes

ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title("Original")
ax1.imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
ax1.set_title("Local Gradient")
ax2.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
ax2.set_title("Markers")
ax3.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax3.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
ax3.set_title("Segmented")

for ax in axes:
    ax.axis('off')

fig.tight_layout()
plt.show()
'''