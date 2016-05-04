import sobelise
import testing_sobel
import numpy as np
from PIL import Image
import glob
import os
import random
from scipy import ndimage
from skimage.transform import rescale, resize
from dwtSliding import dwtSlide 
import matplotlib.pyplot as plt
import water_test

def stack(folderPath,sampleNumber,sobelLevels,brushMasks,superPixMethod='combined',features='combined',triGrown=True,sobelType='combined'):
    FOLDER_PATH = folderPath
    SAMPLE_NUMBER = sampleNumber
    #trainRatio = int(opts['<trainRatio>'])

    #path = './palstaves2/2013T482_Lower_Hardres_Canterbury/Axe1/'
    levels = sobelLevels
    path = FOLDER_PATH
    outputFilename = os.path.join(os.path.dirname(path),'trainingData_'+str(sobelLevels)+'_'+\
    'brush'+str(brushMasks)+'_'+str(superPixMethod)+'_'+str(features)+'_'+'grown'+str(triGrown)+'.npz')
    
    if features=='RGB':
        wholeXArray = np.zeros([0,3])#([0,8+3])#np.zeros([0,levels*3+3])#+3 is for RGB non sobelised 
    elif features=='sobel':
        wholeXArray = np.zeros([0,levels*3+3])    
    elif features=='entropy':
        wholeXArray = np.zeros([0,7+3])
    elif features=='combinedEntSob':
        wholeXArray = np.zeros([0,levels*3+7+3])
    elif features =='dwt':
        wholeXArray = np.zeros([0,8+3])
        #wholeXArray = np.zeros([0,8+3])
    elif features=='combinedDwtSob':
        wholeXArray = np.zeros([0,levels*3+8+3])
        #wholeXArray = np.zeros([0,levels*3+3])
    elif features =='sobelSansRGB':
        wholeXArray = np.zeros([0,levels*3])
    elif features =='sobelHandv':
        wholeXArray = np.zeros([0,levels*6+3])
    
    else: 
        print ("Error selecting type of features")
        l=lp#breakpoint
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
        if counter % 1 == 0:
            if numberSuccessStacked >= trainSetSize or numberStacked == imageSetSize: 
                break
            numberStacked += 1
            numberSuccessStacked += 1
            
            
            
            fileNameStringWithExtension = os.path.basename(filepath)
            fileNameString = os.path.splitext(fileNameStringWithExtension)[0]
            if brushMasks == False:
                maskPath = os.path.join(os.path.dirname(folderPath), 'masks/'+fileNameString+'_mask')##normally 'masks/'
            else:
                maskPath = os.path.join(os.path.dirname(folderPath), 'brushMasks/'+fileNameString+'_mask')##normally 'masks/'
            
            
             # loading 1/4 sized images
            #all levels concatenated together
            #print(maskPath)
            try:
                maskRaw = Image.open(maskPath+'.jpg').convert(mode='L')#convert is new
                imageNameHolder.append(fileNameString)
            except IOError:
                print('Image '+fileNameString+' has no corresponding mask, it has been skipped')
                numberSuccessStacked -= 1
                continue
            
            alreadyDone = sobelise.process_image(filepath,levels,features)
            totalSob = testing_sobel.concatSob(filepath,levels,features)
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
            #im = rescale(im,0.25)
            im = rescale(im,0.25)#0.125)
            if features=='RGB'or features=='entropy'or features=='dwt':
                imArray = im*255
            elif features=='sobel' or features=='sobelHandv'or features=='combinedEntSob'or features=='combinedDwtSob':
                imArray = np.asarray(totalSob)
                imArray = np.dstack([imArray,im*255])
            elif features =='sobelSansRGB':
                imArray = np.asarray(totalSob)
            if features =='entropy'or features =='dwt' or features =='combinedDwtSob' or features =='combinedEntSob':
                dwtFeature = dwtSlide(filepath,4,features)
            flatIm = im.reshape(im.shape[0]*im.shape[1],-1)
            #print(np.max(dwtFeature))
            print(np.max(im))
            #print(np.mean(dwtFeature))
            print(np.mean(im))
            #l=lp
            #l=lp
            #imArray = im
            
            maskArray = np.asarray(maskRaw) #not all 255 or 0 because of compression, may need to threshold
            ## mod the maskArray here
            
            
            if triGrown == True:
                ###maskArray = resize(maskArray,[totalSob.shape[0],totalSob.shape[1]])
                maskArray = resize(maskArray,[im.shape[0],im.shape[1]])
                ##maskArray *= 255
                a=water_test.watershedFunc2(filepath,superPixMethod,trainingSeg=True)
                #print('between')
                featureMap=maskArray
                classifier = 1
                b,totClassified,totMask2,segmentOutlines,totMask=water_test.superPix(im,a,featureMap,classifier,100,alreadyClassified=True,thresh=0.2)
                #print(b.shape)
                maskArray=totMask.reshape([(totClassified.shape[0]),(totClassified.shape[1])])
            else:
                maskArray = resize(maskArray,[im.shape[0],im.shape[1]])
            ###maskArray = ((maskArray+featureMap)/2)
            #plt.imshow(b, interpolation='nearest')
            #plt.show()
            #print(maskArray.shape)
            path2 = os.path.dirname(path)
            newpath=os.path.join(path2,'postTrainingMasks')
            print(newpath)
            if not os.path.exists(newpath):
                
                    os.makedirs(newpath)
            greyRegion=(maskArray == 0.2).astype(int) 
            print(np.max(greyRegion))
            maskImage = Image.fromarray((((maskArray*255).astype(np.uint8))))
            maskImage = maskImage.convert('RGB')
            print(np.array(maskImage).shape)
            #print(np.array(maskImage)[:,:,0])
            #print(np.array(maskImage)[:,:,0])
            maskImage = np.array(maskImage)
            print(np.max(maskImage[...,0]))
            print(np.max(maskImage[...,1]))
            print(np.max(maskImage[...,2]))
            #greyRegion=(maskArray == 0.2*255).astype(int)*1.0
            print(np.max(greyRegion))
            #l=sf
            greyRegionImage = Image.fromarray((((greyRegion*255).astype(np.uint8))))
            greyRegionImage = greyRegionImage.convert('RGB')
            print( np.sum(greyRegion))
            greyRegionImage = np.array(greyRegionImage)
            greyRegionImage[:,:,2:3]=0
            maskImage[:,:,1:3] = 0
            print('hereBrB')
            print( np.sum(greyRegion))
            print( np.max(maskImage))
            print( np.max(greyRegion))
            print(greyRegion.shape)
            print(type(greyRegion))
            #grey255 = greyRegion*255.0
            maskImage2 = (greyRegionImage)+maskImage
            #52 shows red, 51 doesnt
            #superMask2 = (superMask < (a)).astype('int')
            print(np.max(maskImage))
            maskImage = Image.fromarray((((maskImage).astype(np.uint8))))
            print maskImage2.shape
            maskImage2 = Image.fromarray((((maskImage2).astype(np.uint8))))
            
            #maskImage = maskImage.convert('RGB')
            origImage = Image.fromarray((im*255).astype(np.uint8))
            #blend2 = Image.blend(maskImage2,origImage,0.5)
            #maskImage2.save(os.path.join(newpath,fileNameString+'_mask2_training.jpg'))
            greyImage = Image.fromarray((greyRegion*255).astype(np.uint8))
            print im.shape
            #greyImage.save(os.path.join(newpath,fileNameString+'_grey_training.jpg'))
            blend = Image.blend(maskImage2,origImage,0.5)
            blend.save(os.path.join(newpath,fileNameString+'_mask_training.jpg'))
            
            #Image.fromarray((maskArray*255).astype(np.uint8)).save(os.path.join(newpath,fileNameString+'_mask_training.jpg'))
            maskArray = maskArray*255
            flatMaskArray = maskArray.reshape(maskArray.shape[0]*maskArray.shape[1])
            flatImArray = imArray.reshape(imArray.shape[0]*imArray.shape[1],imArray.shape[2])
            #flatIm = np.zeros((flatIm.shape[0],flatIm.shape[1])) # for dwt only testing
            print flatIm.shape
            #print dwtFeature.shape
            if features=='combinedEntSob'or features=='combinedDwtSob' or features=='entropy' or features=='dwt':
                flatImArray = np.hstack([flatImArray,dwtFeature])
            '''
            foreGround = (flatMaskArray>=64)
            backGround = (flatMaskArray<64)
            '''
            
            print( np.max(maskArray))
            foreGround = (flatMaskArray>=100)#values depend on the gray used
            backGround = (flatMaskArray<10)
            
            foreGroundSamples = flatImArray[foreGround,...]
            backGroundSamples = flatImArray[backGround,...]
            print(foreGroundSamples.shape[0])
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
            #print('Stacked image '+fileNameString+ '; number '+str(numberSuccessStacked)+' out of '+str(trainSetSize))
        counter += 1