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

start = time.time()
image = np.asarray(Image.open('C:\Python34\\2013T805_Woolaston_Gloucestershire\SetC\IMG_3751.JPG'))
#image = np.asarray(Image.open(imagePath))
#image = color.rgb2gray(image)
image = rescale(image,0.25)
denoised = gaussian_filter(image, 0)
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
denoised = color.rgb2gray(denoised)
markers = (rank.gradient(denoised, disk(2)) < 10)
markers = ndi.label(markers)[0]
gradient = rank.gradient(denoised, disk(2))
#gradient = np.maximum(np.maximum(np.maximum((rank.gradient(denoisedR, disk(2))),(rank.gradient(denoisedG, disk(2)))),
#(rank.gradient(denoisedB, disk(2)))),rank.gradient(denoised, disk(2)))

# process the watershed
labels = watershed(gradient, markers)
print(np.max(labels))
end = time.time()
print( end - start)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
axes = axes.ravel()
ax0, ax1, ax2, ax3 = axes

ax0.imshow(denoised, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title("Original")
ax1.imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
ax1.set_title("Local Gradient")
ax2.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
ax2.set_title("Markers")
ax3.imshow(denoised, cmap=plt.cm.gray, interpolation='nearest')
ax3.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
ax3.set_title("Segmented")

for ax in axes:
    ax.axis('off')

fig.tight_layout()
plt.show()