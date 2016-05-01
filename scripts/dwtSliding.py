#dfg
import pywt
import numpy as np
from PIL import Image
from skimage.transform import rescale, resize
from skimage.filters.rank import entropy
from skimage.morphology import disk

def dwtSlide(filePath,patchRadius,features):

    im = Image.open(filePath)
    im = np.asarray(im.convert('L'))# * (1.0/255.0)
    #im = np.asarray(im)
    print im.shape
    #l=lp
    im = rescale(im,0.25)#reduce the size of the image for speed
    print("in the dwt")
    
    #print im2.shape
    if features=='entropy'or features=='combinedEntSob':
        coeffStore = entropy(im,disk(10))
        coeffStore = coeffStore.reshape(-1,1)
        coeffStore11 = entropy(im,disk(8))
        coeffStore11 = coeffStore11.reshape(-1,1)
        coeffStore12 = entropy(im,disk(6))
        coeffStore12 = coeffStore12.reshape(-1,1)
        print("in the dwt done first")
        
        coeffStore2 = entropy(im,disk(5))
        coeffStore2 = coeffStore2.reshape(-1,1)
        coeffStore3 = entropy(im,disk(4))
        coeffStore3 = coeffStore3.reshape(-1,1)
        coeffStore4 = entropy(im,disk(3))
        coeffStore4 = coeffStore4.reshape(-1,1)
        coeffStore5 = entropy(im,disk(2))
        coeffStore5 = coeffStore5.reshape(-1,1)
        coeffStore = np.hstack([coeffStore,coeffStore11,coeffStore12,coeffStore2,coeffStore3,coeffStore4,coeffStore5])
    elif features=='dwt'or features=='combinedDwtSob':
    
        imPad = np.pad(im,patchRadius,"symmetric")
        coeffStore = np.zeros((im.shape[0]*im.shape[1],8))
        for i in range(im.shape[0]):#range(1):#range(im.shape[0]):
            print( i )
            if i % 100 ==0 and i>1:
                print coeffVector
                print cA2
                print cH1
            #if i % 400 == 0 and i>1:
            #    l=lp
            for j in range(im.shape[1]):
                #print (j)
                patch = imPad[i:i+patchRadius*2+1,j:j+patchRadius*2+1]
                coeffs = pywt.wavedec2(patch, 'db1', level=1)
                cA2, (cH1, cV1, cD1) = coeffs
                cAMean = np.mean(cA2)
                cAVar = np.var(cA2)
                cH1Mean = np.mean(cH1)
                cH1Var = np.var(cH1)
                cV1Mean = np.mean(cV1)
                cV1Var = np.var(cV1)
                cD1Mean = np.mean(cD1)
                cD1Var = np.var(cD1)
                coeffVector = np.hstack([cAMean,cAVar,cH1Mean,cH1Var,cV1Mean,cV1Var,cD1Mean,cD1Var])
                #coeffVector = np.hstack([np.array(coeffs[0]).reshape(1,-1),np.array(coeffs[1]).reshape(1,-1)])
                ##coeffVector = np.hstack([coeffVector,np.array(coeffs[2]).reshape(1,-1)])
                #print coeffVector
                #if i == 0 and j ==0:
                #    coeffStore = coeffVector
                #else:
                #print j
                #print im.shape
                coeffStore[(i)*im.shape[1]+(j)]=coeffVector #np.vstack([coeffStore,coeffVector])
                #flatMaskArray = maskArray.reshape(maskArray.shape[0]*maskArray.shape[1])
                #X = np.dstack([foreGroundSamples[foreGroundIndices,...],backGroundSamples[backGroundIndices,...]])
        print coeffStore.shape
        print coeffVector
        print np.max(coeffStore[:,0])
        print np.max(coeffStore[:,1])
        print np.max(coeffStore[:,2])
        print np.max(coeffStore[:,3]) 
        print np.mean(coeffStore[:,0])
        print np.mean(coeffStore[:,1])
        print np.mean(coeffStore[:,2])
        print np.mean(coeffStore[:,3])
        
        coeffStore[:,0] = (coeffStore[:,0]-np.mean(coeffStore[:,0]))/np.var(coeffStore[:,0])
        coeffStore[:,1] = (coeffStore[:,1]-np.mean(coeffStore[:,1]))/np.var(coeffStore[:,1])
        coeffStore[:,2] = (coeffStore[:,2]-np.mean(coeffStore[:,2]))/np.var(coeffStore[:,2])
        coeffStore[:,3] = (coeffStore[:,3]-np.mean(coeffStore[:,3]))/np.var(coeffStore[:,3])
        coeffStore[:,4] = (coeffStore[:,4]-np.mean(coeffStore[:,4]))/np.var(coeffStore[:,4])
        coeffStore[:,5] = (coeffStore[:,5]-np.mean(coeffStore[:,5]))/np.var(coeffStore[:,5])
        coeffStore[:,6] = (coeffStore[:,6]-np.mean(coeffStore[:,6]))/np.var(coeffStore[:,6])
        coeffStore[:,7] = (coeffStore[:,7]-np.mean(coeffStore[:,7]))/np.var(coeffStore[:,7])
        
        print np.max(coeffStore[:,0])
        print np.max(coeffStore[:,1])
        print np.max(coeffStore[:,2])
        print np.max(coeffStore[:,3]) 
        print np.var(coeffStore[:,0])
        print np.mean(coeffStore[:,1])
        print np.var(coeffStore[:,2])
        print np.mean(coeffStore[:,3])
        #l=lp    
        #l=lp
        #ab = np.array([[5,4],[2,3]])
        #c = np.pad(ab,2,"symmetric")
        #print(c)
        #print "python2"
        
    
    
    
    #coeffs = pywt.wavedec2(np.ones((8,8)), 'db1', level=2)
    ##cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    #print(cH1)
    return coeffStore
    
    
if __name__ == "__main__":
    dwtSlide(r"C:\Python34\palstaves2\2013T482_Lower_Hardres_Canterbury\Axe4\images\IMG_3304.JPG",4)