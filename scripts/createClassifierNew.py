
import sklearn
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from PIL import Image


#get_ipython().magic('pylab')

def createClassifier(FILE_PATH,CLASSIFIER_TYPE,sobelLevels,trainSample,features,brushMasks,superPixMethod,triGrown):

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
        pickle.dump( classifier, open( os.path.join(dirPath,'LinearSVC'+'_'+str(sobelLevels)+'_'+\
        'brush'+str(brushMasks)+'_'+str(superPixMethod)+'_'+str(features)+'_'+'grown'+str(triGrown)+'.pickle'), "wb" ) )
    elif CLASSIFIER_TYPE == 'Tree':
        from sklearn import tree
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(X,y)
        print (classifier.feature_importances_)
        plt.plot(classifier.feature_importances_)
        plt.savefig(os.path.join(dirPath,'graph_'+str(features)+'.svg'))
        plt.close()
        #l=lp
        pickle.dump( classifier, open( os.path.join(dirPath,'Tree'+'_'+str(sobelLevels)+'_'+\
        'brush'+str(brushMasks)+'_'+str(superPixMethod)+'_'+str(features)+'_'+'grown'+str(triGrown)+'.pickle'), "wb" ) )
    else: 
        print('Classifier: '+CLASSIFIER_TYPE+' has not been recognised')



