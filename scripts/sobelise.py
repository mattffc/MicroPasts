import numpy as np
import docopt
from testing_sobel import concatSob
from skimage.filters import gaussian_filter, sobel_h, sobel_v
from skimage.transform import rescale, resize
from PIL import Image
import os
print('Running Sobelise')
def downsample(I):
    """Downsample I by 2 after blurring with Gaussian of sigma=2.

    Blurs across dimensions 0 and 1. Leaves dimension 2 untouched.

    """
    I = np.atleast_3d(I)
    return rescale(gaussian_filter(I, 2, multichannel=True), 0.5)

def sobel_rgb(I):
    """Like skimage.sobel_{h,v} but works on RGB images. Returns a
    NxMx6 image with channels being hr, hg, hb, vr, vg, vb.

    """
    I = np.atleast_3d(I)
    return np.dstack(
        [sobel_h(I[..., c]) for c in range(I.shape[-1])] +
        [sobel_v(I[..., c]) for c in range(I.shape[-1])]
    )

def process_image(image_fn, levels=1):
    path = os.path.splitext(image_fn)[0]
    counter = 0
    
    im = Image.open(image_fn)
    im = np.asarray(im.convert('RGB')) * (1.0/255.0)
    im = rescale(im,0.25)#reduce the size of the image for speed
    orig_shape = im.shape[:2]
    for level in range(levels):
        print('Saving image layer '+str(counter)+' out of '+str(levels-1))
        counter = 1+counter
        s = resize(sobel_rgb(im), orig_shape)
        s = (255 * (0.5 + s)).astype(np.uint8)
        Image.fromarray(s[..., :3]).save(os.path.join(path+'_'+'{}_h.png'.format(level)))
        Image.fromarray(s[..., 3:]).save(os.path.join(path+'_'+'{}_v.png'.format(level)))
        im = downsample(im) 
    print('Finished all processing')