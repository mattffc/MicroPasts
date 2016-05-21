from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage.morphology import watershed, disk
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
import numpy as np
import random
from PIL import Image
from skimage import color
from skimage.transform import rescale, resize
from skimage.filters import gaussian_filter
import time
from skimage.transform import rescale, resize
from skimage.segmentation import slic
from skimage.data import astronaut
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

image = np.asarray(Image.open(r'C:\Python34\braceletTest\images\IMG_3661.JPG'))
#image = np.asarray(Image.open(imagePath))

#image = color.rgb2gray(image)
image = rescale(image,0.25)

img = image#img_as_float(astronaut()[::2, ::2])
segments_fz = felzenszwalb(img, scale=10, sigma=0.5, min_size=50)
print('done 1')
segments_slic = slic(img, n_segments=250, compactness=10, sigma=1,max_iter=3)
print('done 2')
segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)#kernel_size was 3

print("Felzenszwalb's number of segments: %d" % len(np.unique(segments_fz)))
print("Slic number of segments: %d" % len(np.unique(segments_slic)))
print("Quickshift number of segments: %d" % len(np.unique(segments_quick)))

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
fig.set_size_inches(8, 3, forward=True)
fig.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)

ax[0].imshow(mark_boundaries(img, segments_fz))
ax[0].set_title("Felzenszwalbs's method")
ax[1].imshow(mark_boundaries(img, segments_slic))
ax[1].set_title("SLIC")
ax[2].imshow(mark_boundaries(img, segments_quick))
ax[2].set_title("Quickshift")
for a in ax:
    a.set_xticks(())
    a.set_yticks(())
plt.show()