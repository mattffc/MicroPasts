
import glob

import sobelise
import testing_sobel

import sklearn
import matplotlib
import os
import numpy as np
import pickle
from scipy import ndimage
from PIL import Image
from skimage.transform import rescale, resize
from dwtSliding import dwtSlide 
import water_test
import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
'''
pythonArray = [1,2,3,4]

fileName = "jsonFile.json"

with open(fileName, 'w') as file:
        json.dump(pythonArray, file)

with open(fileName, 'r') as file:
        json = json.load(file)
        print(json)
'''



def useClassifier(FILE_PATH,levels,CLASSIFIER_TYPE,trainSample,superPixMethod,brushMasks):
        path = FILE_PATH
        #levels = 5
        #trainingPath = os.path.join(path,)
        training = np.load(os.path.join(path,'trainingData_'+str(levels)+'_'+str(trainSample)+'.npz'))
        shuffled = training['shuffled']
        trainRatio = training['R']
        #trainingImages = training['header'](images)
        newpath = os.path.join(path,'predictedMasks')
        k=0
        header = training['header'][()]['images']
        searchingFolders = True
        #masksInfo = (levels,CLASSIFIER_TYPE,trainSample,superPixMethod)
        #pickle.dump( masksInfo, open(os.path.join(newpath,"maskInfo.pickle"), "wb" ) )
        while searchingFolders == True:
            
            #print((json.load(open(os.path.join(newpath,"maskInfo.json"), 'rb'))==
            #(levels,CLASSIFIER_TYPE,trainSample,superPixMethod)))
            newpath=os.path.join(path,'predictedMasks')+str(k)
            if os.path.exists(newpath):
                
                    
                    File = open(os.path.join(newpath,"maskInfo.json"), 'r')
                    loadInfo = json.load(File)
                    
                
            #a=pickle.load(open(os.path.join(newpath,"maskInfo.pickle")))
            #print( (loadInfo))
            #print( [levels,CLASSIFIER_TYPE,trainSample,superPixMethod])
            #print((loadInfo==[levels,CLASSIFIER_TYPE,trainSample,superPixMethod]))
            
            #print((loadInfo))
            print((levels,CLASSIFIER_TYPE,trainSample,superPixMethod))
            if not os.path.exists(newpath):
                
                searchingFolders = False
                os.makedirs(newpath)
                masksInfo = {'levels':levels,'CLASSIFIER_TYPE':CLASSIFIER_TYPE,
                'trainSample':trainSample,'superPixMethod':superPixMethod
                ,'brushMasks':brushMasks}
                json.dump( masksInfo, open(os.path.join(newpath,"maskInfo.json"), "w" ) )
                
                json.dump(header,open(os.path.join(newpath,"trainInfo.json"), "w" ))
            elif (loadInfo=={'levels':levels,'CLASSIFIER_TYPE':CLASSIFIER_TYPE,
                'trainSample':trainSample,'superPixMethod':superPixMethod
                ,'brushMasks':brushMasks}):
                searchingFolders = False
            '''
            elif (loadInfo!=(levels,CLASSIFIER_TYPE,trainSample,superPixMethod)):
                #newpath=os.path.join(path,'predictedMasks')+str(k)
                if not os.path.exists(newpath):
                    searchingFolders = False
                    os.makedirs(newpath)
                    masksInfo = (levels,CLASSIFIER_TYPE,trainSample,superPixMethod)
                    json.dump( masksInfo, open(os.path.join(newpath,"maskInfo.json"), "w" ) )
                elif (loadInfo==(levels,CLASSIFIER_TYPE,trainSample,superPixMethod)):
                    searchingFolders = False
            elif (loadInfo==(levels,CLASSIFIER_TYPE,trainSample,superPixMethod)):
                searchingFolders = False
            '''
            k +=1
        print('Sampling rate = '+str(training['S'])+', trainingRatio = '+str(trainRatio))
        #print(training.item())
        header = training['header'][()]['images']
        #print(type(a))
        #print((a[()])['images'])
        #print(header)
        print('Images used as training: '+ str(header))
       
        if CLASSIFIER_TYPE == 'LinearSVC':
            try:
                pickleFile = open(os.path.join(path,'LinearSVC'+'_'+str(levels)+'_'+str(trainSample)+'.pickle'), 'rb')
            except IOError:
                print('Classifier not trained '+'\n'+'##'+'\n'+'##'+'\n'+'##'+'\n'+'##')
                print('##>>>>>>>>'+'\n'+'##>>>>>>>>'+'\n'+'##>>>>>>>>'+'\n'+'##>>>>>>>>')
            classifier = pickle.load(pickleFile)
            
        elif CLASSIFIER_TYPE == 'Tree':
            try:
                pickleFile = open(os.path.join(path,'Tree'+'_'+str(levels)+'_'+str(trainSample)+'.pickle'), 'rb')
            except IOError:
                print('Classifier not trained '+'\n'+'##'+'\n'+'##'+'\n'+'##'+'\n'+'##')
                print('##>>>>>>>>'+'\n'+'##>>>>>>>>'+'\n'+'##>>>>>>>>'+'\n'+'##>>>>>>>>')
            classifier = pickle.load(pickleFile)
            #print(os.path.join(path,'Tree'+'_'+str(levels)+'_'+str(trainSample)+'.pickle'))
            
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
            
            
            #print(imageIndex)
            #print('out of')
            #print(shuffled.shape[0])
            
            
            fileNameStringWithExtension = os.path.basename(filepath)
            fileNameString = os.path.splitext(fileNameStringWithExtension)[0]
            maskPath = os.path.join(path, 'masks/'+fileNameString+'_mask')
            if not os.path.exists(os.path.join(newpath,fileNameString+'_mask.jpg')):
                print('Image '+str(imageIndex+1)+' out of '+str(shuffled.shape[0]+1))
                sobelise.process_image(filepath,levels)
                totalSob = testing_sobel.concatSob(filepath,levels)
                maskMissing = False
                try:
                    maskRaw = Image.open(maskPath+'.jpg')
                    maskMissing = False
                except IOError:
                    print('Image '+fileNameString+' has no corresponding mask, therefore error cannot be calculated')
                    if imageIndex % trainRatio == 0:
                        missingTrain +=1
                    else:
                        missingTest +=1
                    imageIndex += 1
                    maskMissing = True
                    #continue
                
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
                #im = rescale(im,0.25)
                im = rescale(im,0.125)
                #im = im*255 # normalising
                ##imArray = np.asarray(totalSob)
                print(np.mean(totalSob))
                print(np.mean(im))
                #l=lp
                ##imArray = np.dstack([imArray,im*255])
                #new 
                dwtFeature = dwtSlide(filepath,4)
                flatIm = im.reshape(im.shape[0]*im.shape[1],-1)
                flatImArray = np.hstack([flatIm,dwtFeature])
                featureMap = flatImArray
                #print('here b4')
                a=water_test.watershedFunc2(filepath,superPixMethod)
                #print('between')
                b,totClassified,totMask2,segmentOutlines,totMask=water_test.superPix(im,a,featureMap,classifier,100)
                #print('here after')
                print(np.unique((segmentOutlines*255).astype(np.uint8)))
                
                #new end
                #imArray = im
            
                
                #maskArray = np.asarray(maskRaw)
                #maskArray = resize(maskArray,[totalSob.shape[0],totalSob.shape[1]])
                #maskArray *= 255
                #flatMaskArray = maskArray.reshape(maskArray.shape[0]*maskArray.shape[1],1)
                ###flatImArray = imArray.reshape(imArray.shape[0]*imArray.shape[1],imArray.shape[2])
                #predictedMask = classifier.predict(flatImArray)#for superpix
                numberPredicted += 1
                pixelCount = flatImArray.shape[0]
                outputSampleCount = int(1*pixelCount)
                #indices = np.random.choice(pixelCount,replace=False,size=outputSampleCount)
                X = flatImArray#flatImArray[indices,...]
                #y = flatMaskArray#flatMaskArray[indices,...]
                #yPrime = predictedMask.astype(np.int)#for superpix
                
                yPrime = b #new line for superpix
                yPrime = np.asarray(yPrime)
                '''
                totMask2 = np.asarray(totMask2)
                totMask2 = np.reshape(totMask2,(totalSob.shape[0],totalSob.shape[1]))
                totMask2 = rescale(totMask2,4,preserve_range=True)
                totMask2 *= 255
                totMask2 = totMask2.astype(np.uint8)
                '''
                print yPrime.shape
                yPrime = np.reshape(yPrime, (-1, 1)) # -1 means make it whatever it needs to be
                #print(yPrime.shape)
                #print(np.max(yPrime))
                print yPrime.shape
                print im.shape
                yPrimeForMaskSave = np.reshape(yPrime,(im.shape[0],im.shape[1]))
                yPrimeForMaskSaveCopy = np.reshape(yPrime,(im.shape[0],im.shape[1]))
                #print(np.max(yPrimeForMaskSave))
                yPrimeForMaskSave = rescale(yPrimeForMaskSave,8,preserve_range=True,order=0)#order was 1
                #print(np.max(yPrimeForMaskSave))
                #print(np.max(yPrimeForMaskSave))
                yPrimeForMaskSave *= 255
                ##yPrimeForMaskSaveCopy = yPrimeForMaskSaveCopy.astype(np.uint8)
                ##yPrimeForMaskSaveCopy *= 255
                yPrimeForMaskSave = yPrimeForMaskSave.astype(np.uint8)
                ##yPrimeForMaskSaveCopy = yPrimeForMaskSaveCopy.astype(np.uint8)
                #print(os.path.join(newpath,fileNameString+'_mask'))
                if not os.path.exists(os.path.join(path,'preMasks'+str(k-1))):
                    os.makedirs(os.path.join(path,'preMasks'+str(k-1))) 
                basicPath = os.path.join(path,'preMasks'+str(k-1))
                print(np.max(totMask2))
                print(np.min(totMask2))
                
                Image.fromarray(np.uint8(cm.afmhot(totMask2)*255)).save(os.path.join(basicPath,fileNameString+'_ratio_mask.jpg'))
                Image.fromarray(yPrimeForMaskSave).save(os.path.join(newpath,fileNameString+'_mask.jpg'))
                Image.fromarray((totClassified*255).astype(np.uint8)).save(os.path.join(basicPath,fileNameString+'_basic_mask.jpg'))
                segImage = Image.fromarray((((segmentOutlines*255).astype(np.uint8))))
                segImage=segImage.convert('RGB')
                yPrimeForMaskSaveImage = Image.fromarray((((yPrimeForMaskSaveCopy*255).astype(np.uint8))))
                yPrimeForMaskSaveImage=yPrimeForMaskSaveImage.convert('RGB')
                print(np.max(im))
                origImage = Image.fromarray((im*255).astype(np.uint8))
                
                #np.array(segImage)[...,1:3]=0
                #segImage = Image.fromarray((segmentOutlines).astype(np.uint8))
                
                print(np.array(segImage).shape)
                print(np.array(origImage).shape)
                blend = Image.blend(segImage,origImage,0.7)
                blend.save(os.path.join(basicPath,fileNameString+'_segment_mask.jpg'))
                #print(yPrimeForMaskSaveCopy.shape)
                print(im.shape)
                print(segmentOutlines.shape)
                print(yPrimeForMaskSaveCopy.shape)
                blend2 = Image.blend((yPrimeForMaskSaveImage),origImage,0.7)
                blend2.save(os.path.join(basicPath,fileNameString+'_final_mask.jpg'))
                #Image.fromarray((((segmentOutlines*255).astype(np.uint8)))).save(os.path.join(basicPath,fileNameString+'_segment_mask.jpg'))
                #yPrime = (yPrime>64).astype(np.int)
                if maskMissing == False: 
                    maskArray = np.asarray(maskRaw)
                    maskArray = resize(maskArray,[im.shape[0],im.shape[1]])
                    maskArray *= 255
                    flatMaskArray = maskArray.reshape(maskArray.shape[0]*maskArray.shape[1],1)
                    y = flatMaskArray
                    
                    y = (y>64).astype(np.int)
                    absError = (np.absolute(y-yPrime)).sum()
                    print('Error from image '+fileNameString+ ' is '+str(absError))
                
                    if imageIndex % trainRatio == 0:
                        #print('Training Image')
                        totTrainingError = totTrainingError+absError
                    else:
                        totTestingError = totTestingError+absError
                    #totalError = totalError+absError
            else:
                print('Image '+str(imageIndex+1)+' out of '+str(shuffled.shape[0]+1)+' already processed')
                
                try:
                    maskRaw = Image.open(maskPath+'.jpg')
                    maskMissing = False
                except IOError:
                    print('Image '+fileNameString+' has no corresponding mask, therefore error cannot be calculated')
                    if imageIndex % trainRatio == 0:
                        missingTrain +=1
                    else:
                        missingTest +=1
                    imageIndex += 1
                    maskMissing = True
                    continue # was commented
                imgLoad = np.asarray(Image.open(os.path.join(newpath,fileNameString+'_mask.jpg')))
                yPrime = resize(imgLoad,[imgLoad.shape[0]/4,imgLoad.shape[1]/4])
                #yPrime = resize(imgLoad,[imgLoad.shape[0]*imgLoad.shape[1]])
                if (np.max(yPrime) <= 1):
                    yPrime *= 255
                flatYPrime = yPrime.reshape(yPrime.shape[0]*yPrime.shape[1])
                flatYPrime = (flatYPrime>64).astype(np.int)
                flatYPrime = np.reshape(flatYPrime, (-1, 1))
                if maskMissing == False: 
                    maskArray = np.asarray(maskRaw)
                    maskArray = resize(maskArray,[imgLoad.shape[0]/4,imgLoad.shape[1]/4])
                    maskArray *= 255
                    flatMaskArray = maskArray.reshape(maskArray.shape[0]*maskArray.shape[1],1)
                    y = flatMaskArray
                    #print(y.shape)
                    #print(flatYPrime.shape)
                    y = (y>64).astype(np.int)
                    absError = (np.absolute(y-flatYPrime)).sum()
                    print('Error from image '+fileNameString+ ' is '+str(absError))
                
                    if imageIndex % trainRatio == 0:
                        #print('Training Image')
                        totTrainingError = totTrainingError+absError
                    else:
                        totTestingError = totTestingError+absError
                    #totalError = totalError+absError
                 
            #imageIndex += 1
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







