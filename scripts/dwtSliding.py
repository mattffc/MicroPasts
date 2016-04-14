#dfg
import pywt
import numpy as np
from PIL import Image
from skimage.transform import rescale, resize

def dwtSlide(filePath,patchRadius):
    im = Image.open(filePath)
    im = np.asarray(im.convert('L'))# * (1.0/255.0)
    #im = np.asarray(im)
    print im.shape
    #l=lp
    im = rescale(im,0.0625*2)#reduce the size of the image for speed
    
    imPad = np.pad(im,patchRadius,"symmetric")
    coeffStore = np.zeros((im.shape[0]*im.shape[1],100))
    for i in range(im.shape[0]):#range(1):#range(im.shape[0]):
        print( i )
        for j in range(im.shape[1]):
            #print (j)
            patch = imPad[i:i+patchRadius*2+1,j:j+patchRadius*2+1]
            coeffs = pywt.wavedec2(patch, 'db1', level=1)
            #cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
            coeffVector = np.hstack([np.array(coeffs[0]).reshape(1,-1),np.array(coeffs[1]).reshape(1,-1)])
            ##coeffVector = np.hstack([coeffVector,np.array(coeffs[2]).reshape(1,-1)])
            #print coeffVector
            #if i == 0 and j ==0:
            #    coeffStore = coeffVector
            #else:
            #print j
            #print im.shape
            coeffStore[(i+1)*(j+1)-1]=coeffVector #np.vstack([coeffStore,coeffVector])
            #flatMaskArray = maskArray.reshape(maskArray.shape[0]*maskArray.shape[1])
            #X = np.dstack([foreGroundSamples[foreGroundIndices,...],backGroundSamples[backGroundIndices,...]])
    print coeffStore.shape
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