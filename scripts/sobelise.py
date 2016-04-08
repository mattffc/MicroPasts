import numpy as np
#import docopt
from testing_sobel import concatSob
from skimage.filters import gaussian_filter, sobel_h, sobel_v,sobel
from skimage.transform import rescale, resize
from PIL import Image
#from skimage.filters import gaussian_filter
import os
#print('Running Sobelise')
def downsample(I):
    sigma = 2
    """Downsample I by 2 after blurring with Gaussian of sigma=2.

    Blurs across dimensions 0 and 1. Leaves dimension 2 untouched.

    """
    I = np.atleast_3d(I)
    return rescale(gaussian_filter(I, sigma, multichannel=True), 0.5)

def sobel_rgb(I):
    """Like skimage.sobel_{h,v} but works on RGB images. Returns a
    NxMx6 image with channels being hr, hg, hb, vr, vg, vb.

    """
    #denoised = gaussian_filter(image, 2)
    I = np.atleast_3d(I)
    
    return np.dstack(
        [sobel(I[..., c]) for c in range(I.shape[-1])]# +
        #[sobel_v(I[..., c]) for c in range(I.shape[-1])]
    )

def process_image(image_fn, levels=1):
    
    path = os.path.dirname(os.path.dirname(image_fn))
    #print(path)
    if not os.path.exists(os.path.join(path,'sobels')):
        os.makedirs(os.path.join(path,'sobels'))
    counter = 0
    path = os.path.join(path,'sobels')
    im = Image.open(image_fn)
    im = np.asarray(im.convert('RGB')) * (1.0/255.0)
    im = rescale(im,0.25)#reduce the size of the image for speed
    orig_shape = im.shape[:2]
    elseTest = False
    alreadyDone = True
    for level in range(levels):
        saveLocation = os.path.join(path,os.path.splitext((os.path.basename(image_fn)))[0]+'_'+'{}_hv.png'.format(level))
        if not os.path.exists(saveLocation):
            alreadyDone = False
            #print('Saving image layer '+str(counter+1)+' out of '+str(levels))
            counter = 1+counter
          
            s = resize(sobel_rgb(im), orig_shape)
         
            s = (255 * (0.5 + s)).astype(np.uint8)
            #print('here')
            #print((os.path.basename(image_fn)))
            #print(os.path.splitext((os.path.basename(image_fn)))[0])
            #print(os.path.join(path,'_'+'{}_hv.png'.format(level)))
            Image.fromarray(s[..., :3]).save(saveLocation)
            #Image.fromarray(s[..., 3:]).save(os.path.join(path+'_'+'{}_v.png'.format(level)))
            im = downsample(im) 
        else:
            if elseTest == False:
                #print('sobels already exist for ' + str(image_fn))
                elseTest = True
    #print('Finished all processing')
    return alreadyDone
