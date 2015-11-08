
from skimage.transform import rescale, resize
import sklearn
import matplotlib
from PIL import Image
import numpy as np

import glob
import docopt
import os
get_ipython().magic('pylab')

def concatSob(filePath,levels):
	print('blaah')
	filePath = os.path.splitext(filePath)[0]
	totalSob = np.zeros([0,15])
	print(filePath)
	for i in range(levels):
		horzSob = np.asarray(Image.open(os.path.join(filePath+'_'+str(i)+'_h.png')))
		vertSob = np.asarray(Image.open(os.path.join(filePath+'_'+str(i)+'_v.png')))
		horzSobSub = rescale(horzSob, 0.25)
		vertSobSub = rescale(vertSob, 0.25)
		if totalSob.shape[0]==0:
			totalSob = np.zeros([horzSobSub.shape[0],horzSobSub.shape[1]])
		totalSob = np.dstack([totalSob,horzSobSub,vertSobSub])
	totalSob = totalSob[...,1:]	
	return totalSob





