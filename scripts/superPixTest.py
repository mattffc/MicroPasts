import water_test
import numpy as np
import sobelise
import testing_sobel
from skimage.transform import rescale, resize
sobelise.process_image('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3517.JPG',5)
totalSob = testing_sobel.concatSob('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3517.JPG',5)
from PIL import Image #to here done
im = Image.open('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3517.JPG')
im = np.asarray(im)
im = rescale(im,0.25)
imArray = np.asarray(totalSob)
imArray = np.dstack([imArray,im])#to here done
featureMap = imArray
import pickle
pickleFile = open('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\Tree.pickle','rb')
classifier = pickle.load(pickleFile)
a=water_test.watershedFunc2('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3517.JPG')
b=water_test.superPix(im,a,featureMap,classifier,100)