
import sklearn
import matplotlib
import os
import numpy as np
import pickle
from PIL import Image

#get_ipython().magic('pylab')

def createClassifier(FILE_PATH,CLASSIFIER_TYPE,sobelLevels,trainSample):

    path = FILE_PATH

    training = np.load(path)
    dirPath = os.path.dirname(path)
    sampleNumber = training['S']
    trainRatio = training['R']

    X = training['X']
    #print(X.shape)
    
    y = training['y']
    print('Sampling rate = '+str(sampleNumber)+', TrainingRatio = '+str(trainRatio))
    print('Training on approximately '+str(int(X.shape[0]/(sampleNumber*2)))+' images, assuming 18 MegaPixel images and ' +str(sampleNumber)+':1 sampling')

    if CLASSIFIER_TYPE == 'LinearSVC':
        from sklearn.svm import LinearSVC
        classifier = LinearSVC()
        print('Training '+CLASSIFIER_TYPE+' classifier ...')
        classifier.fit(X,y)
        pickle.dump( classifier, open( os.path.join(dirPath,'LinearSVC'+'_'+str(sobelLevels)+'_'+str(trainSample)+'.pickle'), "wb" ) )
    elif CLASSIFIER_TYPE == 'Tree':
        from sklearn import tree
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(X,y)
        pickle.dump( classifier, open( os.path.join(dirPath,'Tree'+'_'+str(sobelLevels)+'_'+str(trainSample)+'.pickle'), "wb" ) )
    else: 
        print('Classifier: '+CLASSIFIER_TYPE+' has not been recognised')



