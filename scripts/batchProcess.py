#sfh
from maskProcess import maskProcess
maskProcess(r'C:\Python34\palstaves2\2013T482_Lower_Hardres_Canterbury\Axe1test\images',
    sobelLevels=5,brushMasks=False,superPixMethod='SLIC',features='sobelSansRGB',triGrown=False,
    sobelType='combined')




