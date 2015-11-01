# coding: utf-8
from PIL import Image
import numpy as np
import sklearn
import matplotlib
get_ipython().magic('pylab')

im = Image.open('C:\\Python34\\palstaves2\\2013T482_Lower_Hardres_Canterbury\\Axe1\\IMG_3530.JPG')
im = np.asarray(im)
Xprime = im.reshape(3456*5184,3)
yprime = classifier.predict(Xprime)
Mask=yprime.reshape(3456,5184)
imshow(Mask)
