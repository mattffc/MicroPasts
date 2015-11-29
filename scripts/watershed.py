import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.morphology import watershed
from PIL import Image
from skimage import color
get_ipython().magic('pylab')


def watershedFunc(sampleNum,image):
    
    sampleNum = 20

    # Generate an initial image with two overlapping circles

    image = np.asarray(Image.open('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3517.JPG'))
    image = color.rgb2gray(image)
    
    print(image.shape[0])
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    samples = np.random.choice(image.shape[0]*image.shape[1],replace=False,size=sampleNum)
    blankMask = np.zeros([image.shape[0]*image.shape[1]])
    blankMask[samples,...] = 1
    blankMask = blankMask.reshape(image.shape[0],image.shape[1])
    markers = ndi.label(blankMask)[0]
    print(blankMask.shape)
    
    labels = watershed(image, markers)
    return labels, markers
    
labels,markers = watershedFunc(1,1)