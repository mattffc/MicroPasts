#sdfg

"""
Stacks images from a folder into a sampled array and saves as an npz.

Usage:
    stackImages.py <folderPath> <sampleNumber> 
    
Options:
    <folderPath>    The path of the folder containing the images.
    <sampleNumber>    The number of samples to take from foreGround and backGround.
    
"""
#<trainRatio> The ratio of test data to training data
import docopt

opts = docopt.docopt(__doc__)

FOLDER_PATH = opts['<folderPath>']
SAMPLE_NUMBER = opts['<sampleNumber>']
#trainRatio = int(opts['<trainRatio>'])

import sobelise
import testing_sobel
import numpy as np
from PIL import Image
import glob
import os
import random
from scipy import ndimage
from skimage.transform import rescale, resize

#path = './palstaves2/2013T482_Lower_Hardres_Canterbury/Axe1/'
levels = 5
path = FOLDER_PATH
outputFilename = os.path.join(path,'trainingData'+str(SAMPLE_NUMBER)+'.npz')
wholeXArray = np.zeros([0,levels*3+3])#+3 is for RGB non sobelised 
wholeyArray = np.zeros([0])
numberStacked = 0
numberSuccessStacked = 0
imageSetSize = 0
trainRatio = 10
get_ipython().magic('run testing_sobel')
get_ipython().magic('run sobelise')
#a=np.load('./palAxe1arrays/JAIMG_3517.npz')
#a=np.asarray(a)
#print(a.shape)

for filepath in glob.glob(os.path.join(path, '*.jpg')):
    imageSetSize += 1
trainSetSize = int(imageSetSize/trainRatio+1)
print('Image set size = '+str(imageSetSize))
print('Training set size = '+str(trainSetSize))
print('Sampling rate = '+str(SAMPLE_NUMBER))
imageNames = glob.glob(os.path.join(path, '*.jpg'))

#random.shuffle(imageNames)
imageNameHolder = []
counter = 0
for filepath in imageNames:
    if counter % 10 == 0:
        if numberSuccessStacked >= trainSetSize or numberStacked == imageSetSize: 
            break
        numberStacked += 1
        numberSuccessStacked += 1
        
        
        
        fileNameStringWithExtension = os.path.basename(filepath)
        fileNameString = os.path.splitext(fileNameStringWithExtension)[0]
        maskPath = os.path.join(path, 'masks/'+fileNameString+'_mask')##normally 'masks/'
        
        sobelise.process_image(filepath,levels)
        totalSob = testing_sobel.concatSob(filepath,levels) # loading 1/4 sized images
        #all levels concatenated together
        
        try:
            maskRaw = Image.open(maskPath+'.jpg').convert(mode='L')#convert is new
            imageNameHolder.append(fileNameString)
        except IOError:
            print('Image '+fileNameString+' has no corresponding mask, it has been skipped')
            numberSuccessStacked -= 1
            continue
        im = Image.open(filepath)
        im = np.asarray(im)
        #im = ndimage.gaussian_filter(im, 3)
        '''
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
        
        maskArray = np.asarray(maskRaw) #not all 255 or 0 because of compression, may need to threshold
        
        maskArray = resize(maskArray,[totalSob.shape[0],totalSob.shape[1]])
        maskArray *= 255
        print(maskArray.shape)
        flatMaskArray = maskArray.reshape(maskArray.shape[0]*maskArray.shape[1])
        flatImArray = imArray.reshape(imArray.shape[0]*imArray.shape[1],imArray.shape[2])
        '''
        foreGround = (flatMaskArray>=64)
        backGround = (flatMaskArray<64)
        '''
        foreGround = (flatMaskArray>=150)#values depend on the gray used
        backGround = (flatMaskArray<20)
        
        foreGroundSamples = flatImArray[foreGround,...]
        backGroundSamples = flatImArray[backGround,...]

        maxSampleCount = min(foreGroundSamples.shape[0],backGroundSamples.shape[0])
        outputSampleCount = min(maxSampleCount,int(SAMPLE_NUMBER))
        foreGroundIndices = np.random.choice(foreGroundSamples.shape[0],replace=False,size=outputSampleCount)
        backGroundIndices = np.random.choice(backGroundSamples.shape[0],replace=False,size=outputSampleCount)
        
        X = np.vstack([foreGroundSamples[foreGroundIndices,...],backGroundSamples[backGroundIndices,...]])
        y = np.concatenate([np.ones(outputSampleCount),np.zeros(outputSampleCount)])
        
        #wholeArray = np.array(joinedArray.shape[0],joinedArray.shape[1])
        #print(wholeArray.shape)
        #a=joinedArray[0:1000000].reshape(4,1000000)
        #print(a.shape)
        wholeXArray = np.concatenate((wholeXArray,X),axis=0)
        wholeyArray = np.concatenate((wholeyArray,y),axis=0)
        header = {'images':imageNameHolder}
        np.savez_compressed(outputFilename,X=wholeXArray,y=wholeyArray,S=int(SAMPLE_NUMBER),R=trainRatio,shuffled=imageNames,header=header)
        print('Stacked image '+fileNameString+ '; number '+str(numberSuccessStacked)+' out of '+str(trainSetSize))
    counter += 1