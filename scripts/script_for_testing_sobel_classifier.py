# coding: utf-8
get_ipython().magic('cd C:\\Users\\Matt\\Documents\\GitHub\\MicroPasts')
get_ipython().magic('cd .\\scripts')
import testing_sobel
import sobelise
import numpy as np
from PIL import Image
from skimage.transform import rescale, resize


#IMG_3566 error = 66000


get_ipython().magic('run sobelise')
#get_ipython().magic('run stackImages C:\\Python34\\palstaves2\\2013T482_Lower_Hardres_Canterbury\\Axe1 100 32')
#get_ipython().magic('run createClassifier C:\\Python34\\palstaves2\\2013T482_Lower_Hardres_Canterbury\\Axe1\\trainingData100.npz Tree')

sobelise.process_image('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3530.JPG',5)




totalSob = testing_sobel.concatSob('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3530.JPG',5)
im = Image.open('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3530.JPG')
im = np.asarray(im)
im = rescale(im,0.25)
imArray = np.dstack([totalSob,im])

flatTotalSob = imArray.reshape(imArray.shape[0]*imArray.shape[1],imArray.shape[2])
predictedTotalSob = classifier.predict(flatTotalSob)

reshapePredicted = predictedTotalSob.reshape(864,1296,1)
im = np.dstack((reshapePredicted,reshapePredicted,reshapePredicted))
imshow(im)
