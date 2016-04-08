"""
Loads classifier and uses it to predict masks and produce and error estimate.

Usage:
    useClassifier.py <filePath> <classifierType>
    
Options:
    <filePath>    The path of the file containing the classifier. [Actually seems to be training data path]
    <classifierType>	The type of classifier to be used
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
import water_test





DIR_PATH = os.path.dirname(FILE_PATH)
def useClassifier():
        path = FILE_PATH
        levels = 5
        #trainingPath = os.path.join(path,)
        training = np.load(path)
        shuffled = training['shuffled']
        trainRatio = training['R']
        #trainingImages = training['header'](images)
        newpath = os.path.join(DIR_PATH,'predictedMasks3')
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            
        print('Sampling rate = '+str(training['S'])+', trainingRatio = '+str(trainRatio))
        #print(training.item())
        header = training['header'][()]['images']
        #print(type(a))
        #print((a[()])['images'])
        #print(header)
        print('Images used as training: '+ str(header))
       
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
        totTestingError = 0
        totTrainingError = 0
        imageSetSize = shuffled.shape[0]
        numberPredicted = 0
        imageIndex = 0
        missingTest = 0
        missingTrain = 0
	
        for filepath in shuffled: #glob.glob(os.path.join(DIR_PATH, '*.jpg')):
            '''
            if imageIndex == int(shuffled.shape[0]/trainRatio+1): 
                averageErrorTraining = totalError/numberPredicted
                print('Average error for training set of '+str(int(shuffled.shape[0]/trainRatio+1))+' images is '+ str(averageErrorTraining))
                totalError = 0
                realTrainSetSize = numberPredicted
            '''
            print('imageIndex')
            print(imageIndex)
            print('out of')
            print(shuffled.shape[0])
            
            
            fileNameStringWithExtension = os.path.basename(filepath)
            fileNameString = os.path.splitext(fileNameStringWithExtension)[0]
            maskPath = os.path.join(DIR_PATH, 'masks/'+fileNameString+'_mask')
           
            sobelise.process_image(filepath,levels)
            totalSob = testing_sobel.concatSob(filepath,levels)
            
            try:
                maskRaw = Image.open(maskPath+'.jpg')
                
            except IOError:
                print('Image '+fileNameString+' has no corresponding mask, it has been skipped')
                if imageIndex % trainRatio == 0:
                    missingTrain +=1
                else:
                    missingTest +=1
                imageIndex += 1
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
            #new 
            featureMap = imArray
            a=water_test.watershedFunc2(filepath)
            b,totClassified,totMask2=water_test.superPix(im,a,featureMap,classifier,100)
            #new end
            #imArray = im
        
            
            maskArray = np.asarray(maskRaw)
            maskArray = resize(maskArray,[totalSob.shape[0],totalSob.shape[1]])
            maskArray *= 255
            flatMaskArray = maskArray.reshape(maskArray.shape[0]*maskArray.shape[1],1)
            flatImArray = imArray.reshape(imArray.shape[0]*imArray.shape[1],imArray.shape[2])
            #predictedMask = classifier.predict(flatImArray)#for superpix
            numberPredicted += 1
            pixelCount = flatImArray.shape[0]
            outputSampleCount = int(1*pixelCount)
            #indices = np.random.choice(pixelCount,replace=False,size=outputSampleCount)
            X = flatImArray#flatImArray[indices,...]
            y = flatMaskArray#flatMaskArray[indices,...]
            #yPrime = predictedMask.astype(np.int)#for superpix
            
            yPrime = b #new line for superpix
            yPrime = np.asarray(yPrime)
            totMask2 = np.asarray(totMask2)
            totMask2 = np.reshape(totMask2,(totalSob.shape[0],totalSob.shape[1]))
            totMask2 = rescale(totMask2,4,preserve_range=True)
            totMask2 *= 255
            totMask2 = totMask2.astype(np.uint8)
            yPrime = np.reshape(yPrime, (-1, 1)) # -1 means make it whatever it needs to be
            print(yPrime.shape)
            print(np.max(yPrime))
            yPrimeForMaskSave = np.reshape(yPrime,(totalSob.shape[0],totalSob.shape[1]))
            print(np.max(yPrimeForMaskSave))
            yPrimeForMaskSave = rescale(yPrimeForMaskSave,4,preserve_range=True)
            print(np.max(yPrimeForMaskSave))
            print(np.max(yPrimeForMaskSave))
            yPrimeForMaskSave *= 255
            yPrimeForMaskSave = yPrimeForMaskSave.astype(np.uint8)
            print(os.path.join(newpath,fileNameString+'_mask'))
            Image.fromarray(totMask2).save(os.path.join(newpath,fileNameString+'_ratio_mask.jpg'))
            Image.fromarray(yPrimeForMaskSave).save(os.path.join(newpath,fileNameString+'_mask.jpg'))
            Image.fromarray((totClassified*255).astype(np.uint8)).save(os.path.join(newpath,fileNameString+'_basic_mask.jpg'))
            #yPrime = (yPrime>64).astype(np.int)
            y = (y>64).astype(np.int)
            absError = (np.absolute(y-yPrime)).sum()
            print('Error from image '+fileNameString+ ' is '+str(absError))
            if imageIndex % trainRatio == 0:
                print('Training Image')
                totTrainingError = totTrainingError+absError
            else:
                totTestingError = totTestingError+absError
            #totalError = totalError+absError
            imageIndex += 1
        '''    
        if imageIndex == int(shuffled.shape[0]/trainRatio): 
            averageErrorTraining = totalError/numberPredicted
            print('Average error for training set of '+str(int(shuffled.shape[0]/trainRatio))+' images is '+ str(averageErrorTraining))
            totalError = 0
            realTrainSetSize = numberPredicted - 1
            averageErrorTest = totalError/(numberPredicted-realTrainSetSize)
            print('Average error for testing set of '+str(imageSetSize-shuffled.shape[0]/trainRatio)+' images is '+ str(averageErrorTest))
        '''
        print('Number Predicted = ' + str(numberPredicted) +' out of '+str(shuffled.shape[0]))
        averageErrorTraining = totTrainingError/(len(header)-missingTrain)
        print('Average error for training set (predicted only) of '+str(int((shuffled.shape[0]/trainRatio+1)-missingTrain))+' images is '+ str(averageErrorTraining))
        averageErrorTest = totTestingError/(shuffled.shape[0]-len(header)-missingTest)
        print('Average error for testing set (predicted only) of '+str((shuffled.shape[0]-len(header)-missingTest))+' images is '+ str(averageErrorTest))
if __name__ == '__main__':
    useClassifier()







