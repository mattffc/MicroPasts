# coding: utf-8
red_mask = im
red_mask[...,1]=0
red_mask[...,2]=0
green_mask = 1-im
np.max(red_mask)
green_mask[...,0]=0
green_mask[...,2]=0
tot_mask = green_mask+red_mask
tot_mask=tot_mask*255
final = image*0.8+tot_mask*0.2
final = final.astype(uint8)
imshow(final)
imshow(green_mask)
imshow(red_mask)
np.max(im)
np.max(green_mask)
np.min(green_mask)
imshow(1-im)
im2 = im*255
green_mask = 255-im2
imshow(green_mask)
imshow(im2)
im2 = im2.astype(uint8)
imshow(im2)
imshow(im)
im = np.dstack((reshapePredicted,reshapePredicted,reshapePredicted))
im2 = im2.astype(uint8)
im2 = im*255
im2 = im2.astype(uint8)
imshow(im2)
