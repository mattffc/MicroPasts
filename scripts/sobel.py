from PIL import Image
import numpy as np
import sklearn
import matplotlib
from scipy import ndimage
get_ipython().magic('pylab')

im = Image.open('C:\\Python34\\palstaves2\\2013T482_Lower_Hardres_Canterbury\\Axe1\\IMG_3520.JPG')
im = np.asarray(im)



sx0 = ndimage.sobel(im[...,...,0], axis=0, mode='constant')
sy0 = ndimage.sobel(im[...,...,0], axis=1, mode='constant')
#sob0 = np.hypot(sx0, sy0)
sx1 = ndimage.sobel(im[...,...,1], axis=0, mode='constant')
sy1 = ndimage.sobel(im[...,...,1], axis=1, mode='constant')
#sob1 = np.hypot(sx1, sy1)
sx2 = ndimage.sobel(im[...,...,2], axis=0, mode='constant')
sy2 = ndimage.sobel(im[...,...,2], axis=1, mode='constant')
#sob2 = np.hypot(sx2, sy2)

sobx = np.dstack([sx0,sx1,sx2])
soby = np.dstack([sy0,sy1,sy2])

sobx_blurred0 = ndimage.gaussian_filter(sobx, 8)
soby_blurred0 = ndimage.gaussian_filter(soby, 8)
#sob_blurred1 = ndimage.gaussian_filter(sob1_3D, 8)
#sob_blurred2 = ndimage.gaussian_filter(sob2_3D, 8)
#sob_blurred = sob_blurred0+sob_blurred1+sob_blurred2
#sob_blurred2 = ndimage.gaussian_filter(sob_blurred, 8)
#sob_blurred3 = ndimage.gaussian_filter(sob_blurred2, 8)
imWithSobBlurred0 = np.dstack([im,sobx,soby,sobx_blurred0,soby_blurred0])
Xprime = imWithSobBlurred0.reshape(3456*5184,15)