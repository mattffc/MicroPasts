"""
Loads classifier and uses it to predict masks and produce and error estimate.

Usage:
    useClassifier.py <filePath> <classifierType>
    
Options:
    <filePath>    The path of the file containing the classifier.
    <classifierType>	The type of classifier to be trained
"""
import glob
import docopt
import sobelise
import testing_sobel

opts = docopt.docopt(__doc__)
FILE_PATH = opts['<filePath>']
CLASSIFIER_TYPE = opts['<classifierType>']

import sklearn
import matplotlib
import os
import numpy as np
import pickle
from scipy import ndimage
from PIL import Image
from skimage.transform import rescale, resize

DIR_PATH = os.path.dirname(FILE_PATH)
def main():
        path = FILE_PATH
        levels = 7
        training = np.load(path)
        shuffled = training['shuffled']
        trainRatio = training['R']
        print('Sampling rate = '+str(training['S'])+', trainingRatio = '+str(trainRatio))
        if CLASSIFIER_TYPE == 'LinearSVC':
            try:
                pickleFile = open(os.path.join(DIR_PATH,'linear-svm.pickle'), 'rb')
            except IOError:
                print('Classifier not trained '+'\n'+'##'+'\n'+'##'+'\n'+'##'+'\n'+'##')
                print('##>>>>>>>>'+'\n'+'##>>>>>>>>'+'\n'+'##>>>>>>>>'+'\n'+'##>>>>>>>>')
            classifier = pickle.load(pickleFile)
        elif CLASSIFIER_TYPE == 'Tree':
            try:
                pickleFile = open(os.path.join(DIR_PATH,'Tree.pickle'), 'rb')
            except IOError:
                print('Classifier not trained '+'\n'+'##'+'\n'+'##'+'\n'+'##'+'\n'+'##')
                print('##>>>>>>>>'+'\n'+'##>>>>>>>>'+'\n'+'##>>>>>>>>'+'\n'+'##>>>>>>>>')
            classifier = pickle.load(pickleFile)
        else:
            print('Classifier requested has not been recognised')
	
        totalError = 0
        imageSetSize = shuffled.shape[0]
        numberPredicted = 0
        imageIndex = 0
	
        for filepath in shuffled: #glob.glob(os.path.join(DIR_PATH, '*.jpg')):
            if imageIndex == int(shuffled.shape[0]/trainRatio): 
                averageErrorTraining = totalError/numberPredicted
                print('Average error for training set of '+str(int(shuffled.shape[0]/trainRatio))+' images is '+ str(averageErrorTraining))
                totalError = 0
                realTrainSetSize = numberPredicted

            imageIndex += 1
            fileNameStringWithExtension = os.path.basename(filepath)
            fileNameString = os.path.splitext(fileNameStringWithExtension)[0]
            maskPath = os.path.join(DIR_PATH, 'masks/'+fileNameString+'_mask')
           
            sobelise.process_image(filepath,levels)
            totalSob = testing_sobel.concatSob(filepath,levels)
            
            try:
                maskRaw = Image.open(maskPath+'.jpg')
            except IOError:
                print('Image '+fileNameString+' has no corresponding mask, it has been skipped')
                continue
            
            im = Image.open(filepath)
            im = np.asarray(im)
            
            '''
            #im = ndimage.gaussian_filter(im, 3)
            
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
            '''
            im = rescale(im,0.25)
            imArray = np.asarray(totalSob)
            imArray = np.dstack([imArray,im])
            #imArray = im
        
            
            maskArray = np.asarray(maskRaw)
            maskArray = resize(maskArray,[totalSob.shape[0],totalSob.shape[1]])
            maskArray *= 255
            flatMaskArray = maskArray.reshape(maskArray.shape[0]*maskArray.shape[1],1)
            flatImArray = imArray.reshape(imArray.shape[0]*imArray.shape[1],imArray.shape[2])
            predictedMask = classifier.predict(flatImArray)
            numberPredicted += 1
            pixelCount = flatImArray.shape[0]
            outputSampleCount = int(1*pixelCount)
            #indices = np.random.choice(pixelCount,replace=False,size=outputSampleCount)
            X = flatImArray#flatImArray[indices,...]
            y = flatMaskArray#flatMaskArray[indices,...]
            yPrime = predictedMask.astype(np.int)#predictedMask[indices,...].astype(np.int)
            yPrime = np.asarray(yPrime)
            yPrime = np.reshape(yPrime, (-1, 1))
            #yPrime = (yPrime>64).astype(np.int)
            y = (y>64).astype(np.int)
            absError = (np.absolute(y-yPrime)).sum()
            print('Error from image '+fileNameString+ ' is '+str(absError))
            totalError = totalError+absError
        if imageIndex == int(shuffled.shape[0]/trainRatio): 
            averageErrorTraining = totalError/numberPredicted
            print('Average error for training set of '+str(int(shuffled.shape[0]/trainRatio))+' images is '+ str(averageErrorTraining))
            totalError = 0
            realTrainSetSize = numberPredicted - 1
            averageErrorTest = totalError/(numberPredicted-realTrainSetSize)
            print('Average error for testing set of '+str(imageSetSize-shuffled.shape[0]/trainRatio)+' images is '+ str(averageErrorTest))

if __name__ == '__main__':
    main()







