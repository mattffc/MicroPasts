# coding: utf-8
get_ipython().magic('cd C:\\Users\\Matt\\Documents\\GitHub\\MicroPasts')
get_ipython().magic('cd .\\scripts')
import testing_sobel
import sobelise






get_ipython().magic('run sobelise')
#get_ipython().magic('run stackImages C:\\Python34\\palstaves2\\2013T482_Lower_Hardres_Canterbury\\Axe1 100 32')
get_ipython().magic('run createClassifier C:\\Python34\\palstaves2\\2013T482_Lower_Hardres_Canterbury\\Axe1\\trainingData100.npz Tree')

sobelise.process_image('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3558.JPG',5)




totalSob = testing_sobel.concatSob('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3558.JPG',5)

flatTotalSob = totalSob.reshape(totalSob.shape[0]*totalSob.shape[1],totalSob.shape[2])
predictedTotalSob = classifier.predict(flatTotalSob)

reshapePredicted = predictedTotalSob.reshape(864,1296,1)
im = np.dstack((reshapePredicted,reshapePredicted,reshapePredicted))
imshow(im)
