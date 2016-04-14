'''
Gather together all scripts for creating masks. This file should take in a directory
for the images and a directory for the masks. It should create a directory for the
finished masks if one doesn't exist. Have a parameter list passed as arguments.
Sobel filters should be stored in a separate folder.

Parameters: 
    Sample pixel number from foreground and background for training: def 10,000
    Sample pixel number from super pixels during classification
    Number of layers of Sobel
    Classification mode; Tree or SVM
    Training mask frequency, default 10
    Compare new masks to true, default False
    Type of super pixeling
    
Output should be average error across produced masks vs test set if it exists.
Masks should go into a folder and create a separate folder for basic masks.    
'''
from stackImagesNew import stack
from createClassifierNew import createClassifier
from useClassifierNew import useClassifier
import os

def maskProcess(folderPath,trainSample=10000,sobelLevels=5,classifier='Tree',
superPixMethod='combined',brushMasks=False):
    if not os.path.exists(os.path.join(os.path.dirname(folderPath),
    'trainingData_'+str(sobelLevels)+'_'+str(trainSample)+'.npz')):
        stack(folderPath,trainSample,sobelLevels,brushMasks,superPixMethod)
        print('Finished preparing training images')
    else:
        print('Training images already prepared')
    if not os.path.exists(os.path.join(os.path.dirname(folderPath),
    classifier+'_'+str(sobelLevels)+'_'+str(trainSample)+'.pickle')):
        createClassifier(os.path.join(os.path.dirname(folderPath),
        'trainingData_'+str(sobelLevels)+'_'+str(trainSample)+'.npz'),classifier,sobelLevels,trainSample)
        print('Finished creating the classifier')
    else:
        print('Classifier already created')
    useClassifier(os.path.dirname(folderPath),sobelLevels,classifier,trainSample,superPixMethod,brushMasks)
    
if __name__ == '__main__':
    maskProcess(r'C:\Python34\palstaves2\2013T482_Lower_Hardres_Canterbury\Axe4\images',
    sobelLevels=5,brushMasks=True,superPixMethod='SLIC')