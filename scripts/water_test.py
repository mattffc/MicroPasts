from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import skimage
from skimage.morphology import watershed, disk
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
import numpy as np
from PIL import Image
from skimage import color
from skimage.transform import rescale, resize
from skimage.filters import gaussian_filter
import time
from skimage.transform import rescale, resize
from skimage.segmentation import slic
from skimage.segmentation import quickshift
from skimage.segmentation import felzenszwalb
from skimage.measure import label

def watershedFunc2(imagePath,superPixMethod,trainingSeg=False):
    start = time.time()
    #image = np.asarray(Image.open('C:\Python34\palstaves2\\2013T482_Lower_Hardres_Canterbury\Axe1\IMG_3517.JPG'))
    image = np.asarray(Image.open(imagePath))
    
    #image = color.rgb2gray(image) --needed for watershed but not slic
    image = rescale(image,0.25)
    #plt.imshow(image)
    #plt.show()
    
    denoised = gaussian_filter(image, 1)# was 2 before
    denoised = color.rgb2gray(denoised)
    # denoise image
    #denoised = rank.median(image, disk(2))
    #plt.imshow(denoised)
    #plt.show()
    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    print(type(denoised))
    #print(denoised)
    denoised -= 0.0000001 # as 1.0 max on denoised was giving errors in the markers
    print(np.max(denoised))
    print(np.min(denoised))
    
    if trainingSeg == False:
        markers = rank.gradient(denoised, disk(6)) < 5 # disk was 2 before, thresh was 10
        
        markers = ndi.label(markers)[0]
        #print(np.max(markers))
        #plt.imshow(gaussian_filter(markers, 4))
        #plt.show()
        #print(np.max(markers))
        # local gradient (disk(2) is used to keep edges thin)
        gradient = rank.gradient(denoised, disk(2))
        #plt.imshow(gradient)
        #plt.show()
        # process the watershed
    
        if superPixMethod == 'combined':
            #print('water start')
            labels = 100000*watershed(gradient, markers)+1
            
            labels += slic(image,max_iter=3,compactness=10,enforce_connectivity=True,min_size_factor=0.01,n_segments=100)
            
            #print('SLIC fin')
        elif superPixMethod == 'watershed':
            labels = watershed(gradient, markers)
        elif superPixMethod == 'SLIC':
            print('here')
            #image = rescale(image,0.5)
            ##labels = felzenszwalb(image, scale=5, sigma=8, min_size=40)
            ##labels = quickshift(image,sigma = 3, kernel_size=3, max_dist=6, ratio=0.5)#kernel size was 3
            #labels = rescale(labels,2)
            
            #labels = 100000*watershed(gradient, markers)
            print('test before errorBBB')
            labels = slic(image,max_iter=5,compactness=10,enforce_connectivity=True,min_size_factor=0.01,n_segments=100)#segements 200 was causing crash,compact was 10
            print('after error')
            #plt.imshow(markers, interpolation='nearest')
            #plt.imshow(labels, interpolation='nearest',cmap="prism")
            #plt.imshow(image, interpolation='nearest', alpha=.8)
            #plt.show()
        else:
            assert(1==2)
    elif trainingSeg == True:
        markers = rank.gradient(denoised, disk(6)) < 5#was 12 disk # disk was 2 before, thresh was 10
        
        markers = ndi.label(markers)[0]
        #print(np.max(markers))
        #plt.imshow(gaussian_filter(markers, 4))
        #plt.show()
        #print(np.max(markers))
        # local gradient (disk(2) is used to keep edges thin)
        gradient = rank.gradient(denoised, disk(2))
        #plt.imshow(gradient)
        #plt.show()
        # process the watershed
        if superPixMethod == 'combined':
            #print('water start')
            labels = 100000*watershed(gradient, markers)+1
            
            labels += slic(image,max_iter=3,compactness=10,enforce_connectivity=True,min_size_factor=0.01,n_segments=100)
            
            #print('SLIC fin')
        elif superPixMethod == 'watershed':
            labels = watershed(gradient, markers)
        elif superPixMethod == 'SLIC':
            print('here')
            #image = rescale(image,0.5)
            ##labels = felzenszwalb(image, scale=5, sigma=8, min_size=40)
            ##labels = quickshift(image,sigma = 3, kernel_size=3, max_dist=6, ratio=0.5)#kernel size was 3
            #labels = rescale(labels,2)
            
            #labels = 100000+watershed(gradient, markers)
            labels = slic(image,max_iter=5,compactness=10,enforce_connectivity=True,min_size_factor=0.01,n_segments=100)#segements 200 was causing crash
            print('here')
            #plt.imshow(markers, interpolation='nearest')
            #plt.imshow(labels, interpolation='nearest',cmap="prism")
            #plt.imshow(image, interpolation='nearest', alpha=.8)
            #plt.show()
        else:
            assert(1==2)
    #print(labels.shape)
    #print(np.max(labels))
    labels = (label(labels, connectivity=1))
    print('Total numer of super pixels = '+str(np.max(labels)))
    
    #print(np.min(labels))
    ###labels = slic(image)
    #plt.imshow(labels)
    #plt.show()
    end = time.time()
    #print(np.max(markers))
    #print( end - start)
    return labels
# display results
def superPix(image,labels,featureMap,classifier,sampleCount=100,alreadyClassified=None,thresh=0.8):
    start = time.time()
    
    
    #print(labels.shape)
    flatLabels = labels.reshape(labels.shape[0]*labels.shape[1])
    i = 1
    totMask = np.zeros([featureMap.shape[0]*featureMap.shape[1]])
    totMask2=np.zeros([featureMap.shape[0]*featureMap.shape[1]])
    if alreadyClassified == None:
        flatFM = featureMap.reshape(featureMap.shape[0]*featureMap.shape[1],featureMap.shape[2])
        totClassified = classifier.predict(flatFM)
    else:
        totClassified = featureMap.reshape([featureMap.shape[0]*featureMap.shape[1]])
    while i < (np.max(flatLabels)+1):
        sampleCount = 100
        if i % 1000 == 0:
            print('Processed '+str(i)+' super pixels out of '+(str(np.max(labels))))
        indices = (flatLabels==i)
        #print(sum(1==indices))
        ###if sum(1==indices) > 0:
        #indices = np.dstack([indices,indices,indices])
        #superPixel = indices*flatImage
        #superPixel = superPixel.reshape(image.shape[0],image.shape[1],image.shape[2])
        ###superPixel = flatFM[indices,...]
        if indices.shape[0]>0:#superPixel.shape[0]>0:
            #print(superPixel.shape[0])
            #print(sampleCount)
            ###sampleCount = min(superPixel.shape[0],sampleCount)
            #print(superPixel.shape)
            ###superPixelInd = np.random.choice(superPixel.shape[0],replace=False,size=sampleCount)
            ###sampSuperPixel = superPixel[superPixelInd,...]
            #totClassified = classifier.predict(flatFM)
            superMask = totClassified[indices,...]#classifier.predict(sampSuperPixel)
            flatZeros = np.zeros(featureMap.shape[0]*featureMap.shape[1])
            #print('indices')
            #print(np.max(indices))
            #print(indices.shape[0])
            
            #print(np.sum(superMask)/superMask.shape[0])
            #print(np.sum(superMask))
            #print(superMask.shape)
            #print(sampSuperPixel.shape)
            #print(superPixelInd)
            #print(superPixel.shape[0])
            #print(sampleCount)
            superMask1 = (superMask > 0.9).astype('int')
            #print(sum(superMask1))
            #print(sum(superMask))
            #print(np.max(superMask))
            #print(superMask.shape)
            #print(superMask < 0.5)
            a=50.0/255.0
            #print(a)
            superMask2 = (superMask < (a)).astype('int')#depends on gray used
            #print(sum(superMask2))
            
            if np.sum(superMask1)/(superMask1.shape[0]*1.0)>thresh:
                k = 0
                j = -1
                '''
                for x in indices:
                    if x == 1:
                        j += 1
                        flatZeros[k,...]=superMask[j]
                    #flatZeros[k,...]=1#superMask[j]
                    k += 1
                '''
                #print(np.sum(flatZeros))
                zeros2d = flatZeros.reshape([featureMap.shape[0],featureMap.shape[1]])    
                #print('in if')
                #print(superPixel.shape)
                #print(featureMap.shape)
                #superPixelMask = (superPixel>0).reshape(featureMap[0],featureMap[1])
                totMask[indices] = 1
            
            elif alreadyClassified == True:#maybe elif here is not logical because if 20% of superpix is white but 80% black we label as white
                #print('sum, then shape')
                #print(sum(superMask2))
                #print(superMask2.shape[0])
                #print(np.sum(superMask2))
                #print(superMask2.shape[0])
                #print(16000/((superMask2.shape[0])))
                
                if np.sum(superMask2)/((superMask2.shape[0]*1.0))>0.1:#should be thresh
                    k = 0
                    j = -1
                    #print('actually BLARGH')
                    #print(np.sum(flatZeros))
                    zeros2d = flatZeros.reshape([featureMap.shape[0],featureMap.shape[1]])    
                    #print('in if')
                    #print(superPixel.shape)
                    #print(featureMap.shape)
                    #superPixelMask = (superPixel>0).reshape(featureMap[0],featureMap[1])
                    totMask[indices] = 0
                else:
                    #print('in else')
                    totMask[indices] = 0.2#125.0/255
                
            
            xlog = (1*np.sum(superMask)/superMask.shape[0])
            ylog = (np.log(xlog+0.00001/(1.00001-xlog))+8)/16
            totMask2[indices] = xlog
            #print(totMask2.shape)
            
        i += 1
    end = time.time()
    print('Time taken on the super pixels = '+ str(end - start))
    totMask = totMask.reshape([featureMap.shape[0],featureMap.shape[1]])
    
    totMask2 = np.asarray(totMask2)
    totMask2 = np.reshape(totMask2,(featureMap.shape[0],featureMap.shape[1]))
    segmentOutlines=skimage.segmentation.find_boundaries(labels)#was totmask2
    
    totMask2 = rescale(totMask2,4,preserve_range=True,order=0)#default is 1 order
    
    totMask2[0:500,0:100]=0.8
    print(np.max(totMask2))
    totMask2 *= 255
    print(np.max(totMask2))
    totMask2 = totMask2.astype(np.uint8)
    print(np.max(totMask2))
    print(np.max(segmentOutlines))
    print(np.min(segmentOutlines))
    ##segmentOutlines = (segmentOutlines*255).astype(np.uint8)
    print(np.max(segmentOutlines))
    print(np.min(segmentOutlines))
    #plt.imshow(segmentOutlines, interpolation='nearest')
    #plt.show()
    labelledRegions = (label(totMask, connectivity=1))
    labelledRegions = np.reshape(labelledRegions,[featureMap.shape[0]*featureMap.shape[1]])
    counts = np.bincount(labelledRegions)
    counts[0]=0
    #print(np.argmax(counts))
    largestRegion = (labelledRegions==np.argmax(counts))
    
    totClassified = totClassified.reshape([featureMap.shape[0],featureMap.shape[1]])
    return largestRegion,totClassified,totMask2,segmentOutlines,totMask#totClassified.reshape(featureMap.shape[0],featureMap.shape[1])
      
'''
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
axes = axes.ravel()
ax0, ax1, ax2, ax3 = axes

ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title("Original")
ax1.imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
ax1.set_title("Local Gradient")
ax2.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
ax2.set_title("Markers")
ax3.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax3.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
ax3.set_title("Segmented")

for ax in axes:
    ax.axis('off')

fig.tight_layout()
plt.show()
'''