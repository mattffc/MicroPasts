# coding: utf-8
get_ipython().magic('cd C:\\Users\\Matt\\Documents\\GitHub\\MicroPasts\\scripts')
import water_test
import numpy as np
a=water_test.watershedFunc2('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3517.JPG')
b=water_test.superPix('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3517.JPG',a)
get_ipython().magic('pylab')
imshow(b)
imshow(a)
get_ipython().magic('run water_test2')
get_ipython().magic('run water_test2')
get_ipython().magic('run water_test2')
get_ipython().magic('run water_test2')
get_ipython().magic('run water_test2')
get_ipython().magic('run water_test2')
get_ipython().magic('run water_test2')
get_ipython().magic('run water_test2')
get_ipython().magic('run water_test2')
get_ipython().magic('run water_test2')
import sobelise
import testing_sobel
totalSob = testing_sobel.concatSob('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3517.JPG',8)
from PIL import Image
im = Image.open('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3517.JPG')
im = np.asarray(im)
im = rescale(im,0.25)
imArray = np.asarray(totalSob)
imArray = np.dstack([imArray,im])
featureMap = imArray
import pickle
classifier = pickle.load(C:\Python34\palstaves2\2013T482_Lower_Hardres_Canterbury\Axe1\Tree.pickle)
classifier = pickle.load('C:\Python34\palstaves2\2013T482_Lower_Hardres_Canterbury\Axe1\Tree.pickle')
classifier = pickle.load('C:\Python34\palstaves2\2013T482_Lower_Hardres_Canterbury\Axe1\Tree.pickle','rb')
pickleFile = open('C:\Python34\palstaves2\2013T482_Lower_Hardres_Canterbury\Axe1','rb')
pickleFile = open('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1','rb')
pickleFile = open('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\Tree.pickle','rb')
classifier = pickle.load(pickleFile)
a=water_test.watershedFunc2('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3517.JPG')
b=water_test.superPix(a,featureMap,classifier)
b=water_test.superPix(a,featureMap,classifier)
import water_test
b=water_test.superPix(a,featureMap,classifier)
