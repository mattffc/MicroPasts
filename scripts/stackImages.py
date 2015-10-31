#sdfg

"""
Stacks images from a folder into a sampled array and saves as an npz.

Usage:
    stackImages.py <folderPath> 
    
Options:
    <folderPath>    The path of the folder containing the images.
"""
import docopt

opts = docopt.docopt(__doc__)

FOLDER_PATH = opts['<folderPath>']

import numpy as np
from PIL import Image
import glob
import os
import random

#path = './palstaves2/2013T482_Lower_Hardres_Canterbury/Axe1/'
path = FOLDER_PATH
outputFilename = os.path.join(path,'trainingData0001.npz')
wholeXArray = np.zeros([0,3])
wholeyArray = np.zeros([0])
numberStacked = 0
imageSetSize = 0
samplingRate = 0.0001 # normally 0.01
#a=np.load('./palAxe1arrays/JAIMG_3517.npz')
#a=np.asarray(a)
#print(a.shape)

for filepath in glob.glob(os.path.join(path, '*.jpg')):
	imageSetSize += 1
trainSetSize = int(imageSetSize/4)
print('Image set size = '+str(imageSetSize))
print('Training set size = '+str(trainSetSize))
print('Sampling rate = '+str(samplingRate))
shuffled = glob.glob(os.path.join(path, '*.jpg'))
random.shuffle(shuffled)
for filepath in shuffled:
	if numberStacked >= trainSetSize: 
		break
	numberStacked += 1
	fileNameStringWithExtension = os.path.basename(filepath)
	fileNameString = os.path.splitext(fileNameStringWithExtension)[0]
	maskPath = os.path.join(path, 'masks/'+fileNameString+'_mask')
	
	try:
		maskRaw = Image.open(maskPath+'.jpg')
	except IOError:
		print('Image '+fileNameString+' has no corresponding mask, it has been skipped')
		continue
	imRaw = Image.open(filepath)
	imArray = np.asarray(imRaw)
	print(filepath)
	maskArray = np.asarray(maskRaw) #not all 255 or 0 because of compression, may need to threshold
	flatMaskArray = maskArray.reshape(maskArray.shape[0]*maskArray.shape[1])
	flatImArray = imArray.reshape(imArray.shape[0]*imArray.shape[1],3)
	
	foreGround = (flatMaskArray>=64)
	backGround = (flatMaskArray<64)
	foreGroundSamples = flatImArray[foreGround,...]
	backGroundSamples = flatImArray[backGround,...]

	maxSampleCount = min(foreGroundSamples.shape[0],backGroundSamples.shape[0])
	outputSampleCount = min(maxSampleCount,1000)
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
	np.savez_compressed(outputFilename,X=wholeXArray,y=wholeyArray)	
	print('Stacked image number '+str(numberStacked)+' out of '+str(trainSetSize))
