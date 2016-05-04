#sfh
from maskProcess import maskProcess
try: 
    maskProcess(r'C:\Python34\bellTest\images',classifier='LinearSVC',
        sobelLevels=5,brushMasks=True,superPixMethod='None',features='RGB',triGrown=False)
except:
    print'was not good'
try:    
    maskProcess(r'C:\Python34\bellTest\images',classifier='Tree',
        sobelLevels=5,brushMasks=True,superPixMethod='None',features='RGB',triGrown=False)
except:
    print'was not good'
try:     
    maskProcess(r'C:\Python34\bellTest\images',
        sobelLevels=1,brushMasks=True,superPixMethod='None',features='sobel',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\bellTest\images',
        sobelLevels=3,brushMasks=True,superPixMethod='None',features='sobel',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\bellTest\images',
        sobelLevels=5,brushMasks=True,superPixMethod='None',features='sobel',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\bellTest\images',
        sobelLevels=7,brushMasks=True,superPixMethod='None',features='sobel',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\bellTest\images',
        sobelLevels=5,brushMasks=True,superPixMethod='None',features='sobelHandv',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\bellTest\images',
        sobelLevels=5,brushMasks=True,superPixMethod='None',features='sobelSansRGB',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\bellTest\images',
        sobelLevels=5,brushMasks=True,superPixMethod='None',features='combinedEntSob',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\bellTest\images',
        sobelLevels=5,brushMasks=True,superPixMethod='None',features='entropy',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\bellTest\images',
        sobelLevels=5,brushMasks=True,superPixMethod='watershed',features='combinedEntSob',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\bellTest\images',
        sobelLevels=5,brushMasks=True,superPixMethod='SLIC',features='combinedEntSob',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\bellTest\images',
        sobelLevels=5,brushMasks=True,superPixMethod='combined',features='combinedEntSob',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\bellTest\images',
        sobelLevels=5,brushMasks=True,superPixMethod='SLIC',features='combinedEntSob',triGrown=True)
except:
    print'was not good'    
    


    

try: 
    maskProcess(r'C:\Python34\stonecrossTest\images',classifier='LinearSVC',
        sobelLevels=5,brushMasks=False,superPixMethod='None',features='RGB',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\stonecrossTest\images',classifier='Tree',
        sobelLevels=5,brushMasks=False,superPixMethod='None',features='RGB',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\stonecrossTest\images',
        sobelLevels=1,brushMasks=False,superPixMethod='None',features='sobel',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\stonecrossTest\images',
        sobelLevels=3,brushMasks=False,superPixMethod='None',features='sobel',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\stonecrossTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='None',features='sobel',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\stonecrossTest\images',
        sobelLevels=7,brushMasks=False,superPixMethod='None',features='sobel',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\stonecrossTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='None',features='sobelHandv',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\stonecrossTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='None',features='sobelSansRGB',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\stonecrossTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='None',features='combinedEntSob',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\stonecrossTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='None',features='entropy',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\stonecrossTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='watershed',features='combinedEntSob',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\stonecrossTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='SLIC',features='combinedEntSob',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\stonecrossTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='combined',features='combinedEntSob',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\stonecrossTest\images',
        sobelLevels=5,brushMasks=True,superPixMethod='SLIC',features='combinedEntSob',triGrown=False)    
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\stonecrossTest\images',
        sobelLevels=5,brushMasks=True,superPixMethod='SLIC',features='combinedEntSob',triGrown=True)    
except:
    print'was not good'   
    


try: 
    maskProcess(r'C:\Python34\braceletTest\images',classifier='LinearSVC',
        sobelLevels=5,brushMasks=False,superPixMethod='None',features='RGB',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\braceletTest\images',classifier='Tree',
        sobelLevels=5,brushMasks=False,superPixMethod='None',features='RGB',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\braceletTest\images',
        sobelLevels=1,brushMasks=False,superPixMethod='None',features='sobel',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\braceletTest\images',
        sobelLevels=3,brushMasks=False,superPixMethod='None',features='sobel',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\braceletTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='None',features='sobel',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\braceletTest\images',
        sobelLevels=7,brushMasks=False,superPixMethod='None',features='sobel',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\braceletTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='None',features='sobelHandv',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\braceletTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='None',features='sobelSansRGB',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\braceletTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='None',features='combinedEntSob',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\braceletTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='None',features='entropy',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\braceletTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='watershed',features='combinedEntSob',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\braceletTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='SLIC',features='combinedEntSob',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\braceletTest\images',
        sobelLevels=5,brushMasks=False,superPixMethod='combined',features='combinedEntSob',triGrown=False)
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\braceletTest\images',
        sobelLevels=5,brushMasks=True,superPixMethod='SLIC',features='combinedEntSob',triGrown=False)    
except:
    print'was not good'
try: 
    maskProcess(r'C:\Python34\braceletTest\images',
        sobelLevels=5,brushMasks=True,superPixMethod='SLIC',features='combinedEntSob',triGrown=True) 
except:
    print'was not good'
