
from skimage.transform import rescale, resize
import sklearn
import matplotlib
from PIL import Image
import numpy as np

import glob
#import docopt
import os
get_ipython().magic('pylab')

def concatSob(filePath,levels=1,features='combinedEntSob'):
     
     #print('Concatinating saved sobel images...')
     filePath = os.path.splitext(filePath)[0]
     imageName = os.path.basename(filePath)
     filePath= os.path.dirname(os.path.dirname(filePath))
     filePath = os.path.join(filePath,'sobels',imageName)
     #print('concat filepath')
     #print(filePath)
     totalSob = np.zeros([0,15])
     #print(filePath)
     if not features == 'sobelHandv':
         for i in range(levels):
              #print('Concatinating level '+str(i))
              #horzSob = np.asarray(Image.open(os.path.join(filePath+'_'+str(i)+'_h.png')))
              #vertSob = np.asarray(Image.open(os.path.join(filePath+'_'+str(i)+'_v.png')))
              hvSob = np.asarray(Image.open(os.path.join(filePath+'_'+str(i)+'_hv.png')))
              #horzSobSub = rescale(horzSob, 0.25)
              #vertSobSub = rescale(vertSob, 0.25)
              if totalSob.shape[0]==0:
                       print hvSob.shape
                       totalSob = np.zeros([hvSob.shape[0],hvSob.shape[1]])
              totalSob = np.dstack([totalSob,hvSob])
         totalSob = totalSob[...,1:]	
         #print('Finished concatinating')
     elif features == 'sobelHandv':
        for i in range(levels):
              #print('Concatinating level '+str(i))
              horzSob = np.asarray(Image.open(os.path.join(filePath+'_'+str(i)+'_h.png')))
              vertSob = np.asarray(Image.open(os.path.join(filePath+'_'+str(i)+'_v.png')))
              #hvSob = np.asarray(Image.open(os.path.join(filePath+'_'+str(i)+'_hv.png')))
              #horzSobSub = rescale(horzSob, 0.25)
              #vertSobSub = rescale(vertSob, 0.25)
              if totalSob.shape[0]==0:
                       #print hvSob.shape
                       totalSob = np.zeros([horzSob.shape[0],horzSob.shape[1]])
              totalSob = np.dstack([totalSob,horzSob,vertSob])
        totalSob = totalSob[...,1:]	
        #print('Finished concatinating')
     else:
        print('Sobeltype not understood')
        l=lp
     return totalSob





