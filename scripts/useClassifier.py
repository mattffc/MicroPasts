"""
Loads classifier and uses it to predict masks and produce and error estimate.

Usage:
    useClassifier.py <folderPath> <classifierType>
    
Options:
    <folderPath>    The path of the folder containing the classifier.
    <classifierType>	The type of classifier to be trained
"""
import glob
import docopt

opts = docopt.docopt(__doc__)
FOLDER_PATH = opts['<folderPath>']
CLASSIFIER_TYPE = opts['<classifierType>']
import sklearn
import matplotlib
import os
import numpy as np
import pickle
from PIL import Image

def main():
	path = FOLDER_PATH
	if CLASSIFIER_TYPE == 'LinearSVC':
		try:
			pickleFile = open(os.path.join(path,'linear-svm.pickle'), 'rb')
		except IOError:
			print('Classifier is valid but has not been trained yet')
			
		classifier = pickle.load(pickleFile)
	else:
		print('Classifier requested has not been recognised')
		
	totalError = 0
	imageSetSize = 0
	numberPredicted = 0
	imageIndex = 0

	for filepath in glob.glob(os.path.join(path, '*.jpg')):
		imageSetSize += 1
		trainSetSize = int(imageSetSize/2)
	print('Image set size = '+str(imageSetSize))
	print('Training set size = '+str(trainSetSize))

	for filepath in glob.glob(os.path.join(path, '*.jpg')):
		if imageIndex == trainSetSize: 
			averageErrorTraining = totalError/numberPredicted
			print('Average error for training set of '+str(trainSetSize)+' images is '+ str(averageErrorTraining))
			totalError = 0
			realTrainSetSize = numberPredicted

		imageIndex += 1
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
		maskArray = np.asarray(maskRaw)
		flatMaskArray = maskArray.reshape(maskArray.shape[0]*maskArray.shape[1],1)
		flatImArray = imArray.reshape(imArray.shape[0]*imArray.shape[1],3)
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
	averageErrorTest = totalError/(numberPredicted-realTrainSetSize)
	print('Average error for testing set of '+str(imageSetSize-trainSetSize)+' images is '+ str(averageErrorTest))

if __name__ == '__main__':
    main()







