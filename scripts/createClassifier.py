"""
Loads training data and uses it to create a classifier.

Usage:
    createClassifier.py <folderPath> <classifierType>
    
Options:
    <folderPath>    The path of the folder containing the training data.
    <classifierType>	The type of classifier to be trained

"""
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

get_ipython().magic('pylab')

path = FOLDER_PATH
training = np.load(os.path.join(path,'trainingData0001.npz'))

X = training['X']
y = training['y']
print('training on approximately '+str(int(X.shape[0]/1800))+' images, assuming 18MegaPixel images and 10000:1 sampling')

if CLASSIFIER_TYPE == 'LinearSVC':
	from sklearn.svm import LinearSVC
	classifier = LinearSVC()
	print('Training '+CLASSIFIER_TYPE+' classifier ...')
	classifier.fit(X,y)
	
	
else:
	print('Classifier: '+CLASSIFIER_TYPE+' has not been recognised')

pickle.dump( classifier, open( os.path.join(path,"linear-svm6.pickle"), "wb" ) )

