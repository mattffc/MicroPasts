
"""
Loads sobel images and uses them to predict mask.

Usage:
    testing_sobel.py <filePath> <levels>
    
Options:
    <filePath>    The path of the base file of the sobel images.
	<levels>	The number of levels created by process_image (downsampling levels)
    
"""
from skimage.transform import rescale, resize
import sklearn
import matplotlib
from PIL import Image
import numpy as np

import glob
import docopt
import os
get_ipython().magic('pylab')

opts = docopt.docopt(__doc__)
FILE_PATH = opts['<filePath>']
LEVELS = int(opts['<levels>'])

totalSob = np.zeros([0,15])

for i in range(LEVELS):
	horzSob = np.asarray(Image.open(os.path.join(FILE_PATH+'_'+str(i)+'_h.png')))
	vertSob = np.asarray(Image.open(os.path.join(FILE_PATH+'_'+str(i)+'_v.png')))
	horzSobSub = rescale(horzSob, 0.25)
	vertSobSub = rescale(vertSob, 0.25)
	if totalSob.shape[0]==0:
		totalSob = np.zeros([horzSobSub.shape[0],horzSobSub.shape[1]])
	totalSob = np.dstack([totalSob,horzSobSub,vertSobSub])
	
totalSob = totalSob[...,1:]





