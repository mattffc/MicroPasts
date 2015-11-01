"""
Loads training data and uses it to create a classifier.

Usage:
    createClassifier.py <filePath> <classifierType>
    
Options:
    <filePath>    The path of the file containing the training data.
    <classifierType>	The type of classifier to be trained

"""
import docopt

opts = docopt.docopt(__doc__)
FILE_PATH = opts['<filePath>']
CLASSIFIER_TYPE = opts['<classifierType>']
import sklearn
import matplotlib
import os
import numpy as np
import pickle
from PIL import Image

get_ipython().magic('pylab')

path = FILE_PATH

training = np.load(path)
dirPath = os.path.dirname(path)
sampleNumber = training['S']
trainRatio = training['R']

X = training['X']
y = training['y']
print('Sampling rate = '+str(sampleNumber)+', TrainingRatio = '+str(trainRatio))
print('training on approximately '+str(int(X.shape[0]/(sampleNumber*2)))+' images, assuming 18MegaPixel images and ' +str(sampleNumber)+':1 sampling')

if CLASSIFIER_TYPE == 'LinearSVC':
	from sklearn.svm import LinearSVC
	classifier = LinearSVC()
	print('Training '+CLASSIFIER_TYPE+' classifier ...')
	classifier.fit(X,y)
	pickle.dump( classifier, open( os.path.join(dirPath,"linear-svm.pickle"), "wb" ) )
elif CLASSIFIER_TYPE == 'Tree':
	from sklearn import tree
	classifier = tree.DecisionTreeClassifier()
	classifier.fit(X,y)
	pickle.dump( classifier, open( os.path.join(dirPath,"Tree.pickle"), "wb" ) )
else:
	print('Classifier: '+CLASSIFIER_TYPE+' has not been recognised')



