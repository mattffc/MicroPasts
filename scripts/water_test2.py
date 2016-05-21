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

start = time.time()
image = np.asarray(Image.open(r'C:\Python34\stonecrossTest2\images\IMG_8502.JPG'))
#image = np.asarray(Image.open(imagePath))

#image = color.rgb2gray(image)
image = rescale(image,0.25)
#image = (image*255).astype(int)
denoised = gaussian_filter(image, 1)
denoisedR = denoised[...,0]
denoisedG = denoised[...,1]
denoisedB = np.asarray(denoised[...,2])
denoisedB -= 0.000000000001
# denoise image
#denoised = rank.median(image, disk(2))

# find continuous region (low gradient -
# where less than 10 for this image) --> markers
# disk(5) is used here to get a more smooth image

print(np.max(denoisedB))
print(np.max(denoisedG))
print(np.max(denoisedR))
c = (rank.gradient(denoisedB, disk(2)))
print(np.max(denoisedB))
print(denoisedG[5,5])
print(np.max(((np.maximum((rank.gradient(denoisedR, disk(2))),(rank.gradient(denoisedG, disk(2))))))))
print(np.max((rank.gradient(denoisedB, disk(2)))))
#markers = (np.maximum(np.maximum((rank.gradient(denoisedR, disk(1))),(rank.gradient(denoisedG, disk(1)))),
#(rank.gradient(denoisedB, disk(1)))) < 10)


# local gradient (disk(2) is used to keep edges thin)
denoised = color.rgb2gray(denoised) # denisoed disk 2
markers = (rank.gradient(denoised, disk(3))<8)#disk(3))<10)#disk(2)) < 18) # disk 4 works well with thresh 8 and guass 2 # try disk 5 <10
markers = ndi.label(markers)[0]
#markers = markers*np.random.rand(markers.shape[0],markers.shape[1])
print('npmax markers')
print(np.max(markers))
#markers = random.shuffle(markers)
gradient = rank.gradient(denoised, disk(2))
#gradient = np.maximum(np.maximum(np.maximum((rank.gradient(denoisedR, disk(2))),(rank.gradient(denoisedG, disk(2)))),
#(rank.gradient(denoisedB, disk(2)))),rank.gradient(denoised, disk(2)))

# process the watershed
labels = 2*watershed(gradient, markers)+1

##labels += 2*slic(image,max_iter=3,compactness=10,enforce_connectivity=True,min_size_factor=0.01,n_segments=200)
print(np.sum(labels))
'''
for i in range(np.max(labels)+1):
    indicies = np.where(labels==i)
    #print((np.random.rand()))
    labels[indicies] *= np.random.rand()
'''
print(np.max(labels))
print(np.min(labels))
print(np.sum(labels))
end = time.time()
print( end - start)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
axes = axes.ravel()
ax0, ax1, ax2, ax3 = axes

ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title("Original")
ax1.imshow(gradient, cmap = plt.cm.Greys_r, interpolation='nearest')
ax1.set_title("Local Gradient")
ax2.imshow(markers, cmap=plt.cm.prism, interpolation='nearest')
ax2.set_title("Markers")
ax3.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
#labels = labels*np.random.rand(labels.shape[0],labels.shape[1])
#labels = random.shuffle(labels)
ax3.imshow(labels, cmap=plt.cm.prism, interpolation='nearest', alpha=.99)
ax3.set_title("Segmented")

for ax in axes:
    ax.axis('off')

fig.tight_layout()
plt.show()