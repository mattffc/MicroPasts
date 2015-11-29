'''
Region growing algorithm for image segmentation
Written by Matthew ffrench-Constant 18/09/2014
Based on a paper by Frank Shih and Shouxian Cheng called 'Automatic seeded region growing for color image segmentation' 2004
'''



import sys
import numpy as np
import cv2
import math
import random
import bisect
import copy




#-------------------------------------------------------------------------------------------------
# Variables

img_name = 'landscape.jpg' #name of image to segment
max_regions = 4 #number of regions at which region merging will stop
otsu_factor = 2.5 #fiddle factor for deciding seed locations from Otsu's method, it shifts the otsu threshold. Higher value, fewer seeds
initial_min_region_size = 50 #number of pixels within a region below which it will be merged
region_growth_factor = 1.25 #rate at which the min_region_size is increased after each iteration of merging
colour_space = 'RGB' #choose from BGR, YCbCr, LAB
euclid_bias = 2000 #fiddle factor used in the calculation for Euclidean distance between pixels and regions. Equation that uses this is dubious - needs looking into
euclid_region_thresh = 0.015 #threshold for determining whether to merge regions based on mean colour similarity

# Compile variables into a dictionary

variable_list = {'img_name':img_name,'max_regions':max_regions,'otsu_factor':otsu_factor,\
                'initial_min_region_size':initial_min_region_size,\
                'region_growth_factor':region_growth_factor,'colour_space':colour_space,\
                'euclid_bias':euclid_bias,'euclid_region_thresh':euclid_region_thresh}




#-------------------------------------------------------------------------------------------------
# Main function which calls all other functions

def Main(variable_list):

  img = cv2.imread(variable_list['img_name'],cv2.IMREAD_COLOR)
  img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC) # resize image, generally use 0.5 or 0.25 as fullsize takes about 5 mins
  img2 = np.array(img) # make copies of img for later use
  img3 = np.array(img)

  img = convert_colourspace(img,variable_list['colour_space']) # convert colour_space

  ''' Reference: Shih page 2, equation (2) concerning standard deviation - note "Condition 2" on page 3 not used'''
  img = np.float32(img) #temporarily convert to float for calculating standard deviation
  std_dev_img = create_std_dev_img(img) #create standard deviation image using the difference of two filtered images
  img = np.uint8(img) #convert image back to uint8

  contours, hierarchy = threshold_and_find_contours(img2,std_dev_img) # call function to find contours of thresholded image ie seed regions

  region_colour_store = [] # initialise colour store
  mask_sum, region_colour_store = label_seed_regions(img,contours,hierarchy,region_colour_store) # label regions in matrix 'mask_sum',also store mean colour

  ''' Reference: Shih page 4, concerning the creation of an array of adjacent pixels called T'''
  Tprime = initialise_Tprime(mask_sum) # add pixels adjacent to seeds to array 'Tprime', Tprime only contains pixel coords
  T = [] # initialise T
  T,Tprime  = Tprime_to_T(T,Tprime,img, mask_sum, region_colour_store) # convert Tprime to T, 'T' contains pixel coords and euclidean distances
  T = sorted(T, key=lambda tup: tup[3]) # sort T by euclidean distance

  while len(T)>0: # iterate until T is empty, at this point all pixels have been classified
    T,Tprime,mask_sum  = Add_T_element_to_region(T,mask_sum) # add T element to region, update: T, Tprime and mask_sum
    T,Tprime  = Tprime_to_T(T,Tprime,img, mask_sum, region_colour_store) # convert new Tprime pixels to T

  mask_sum_hold = np.uint8(mask_sum) # convert mask_sum aka label matrix, to uint8 so it can be displayed
  cv2.imshow('image',(mask_sum_hold*10)) # show mask_sum's labelled areas
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  ''' Reference: Shih page 5, concerning rules for region merging'''
  mask_sum = region_merging(variable_list, mask_sum, region_colour_store) # perform region merging on mask_sum
  cv2.imshow('image',(mask_sum*10)) # display mask_sum after region merging
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  contours = collate_contours(mask_sum,img3) # find the contours of each labelled region and display each image
  cv2.imshow('image',(img3)) # display the original image
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.drawContours(img3, contours, -1, (0,0,255), 1) # draw the contours found in 'collate_contours' onto the original img
  cv2.imshow('image',(img3)) # show the original image with contours drawn on
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.imwrite('cheng2_2img.jpg',img3)

#-------------------------------------------------------------------------------------------------

def region_merging(variable_list, mask_sum, region_colour_store):

  mask_sum = mask_sum[:,:,0] # mask_sum is 3 deep but labels only stored in zeroth plane

  # start looping through each region
  region_store = np.unique(mask_sum) # extract the indicies of the regions in mask_sum
  number_of_regions = len(region_store) # calculate number of regions

  number_of_regions_end = copy.copy(number_of_regions)
  number_of_regions_start = copy.copy(number_of_regions)


  thresh4_area = variable_list['initial_min_region_size']
  while number_of_regions_end > variable_list['max_regions']:
    number_of_regions_start = len(region_store)

    for i in region_store:

      if number_of_regions_end > variable_list['max_regions']:
        current_region_index = i
        truth_mask = localise_region(mask_sum,current_region_index)
        hold_truth_mask = []
        hold_truth_mask[:] = truth_mask
        hold_truth_mask = np.array(hold_truth_mask)
        truth_mask_contour_area = (np.nonzero(truth_mask)[0].shape)[0]
        adjacent_regions = find_adjacent_regions(mask_sum,truth_mask)
        current_zone_colour_index = ([x for x, y in enumerate(region_colour_store) if y[0] == current_region_index])[0]
        current_zone_colour = region_colour_store[current_zone_colour_index][1]
        region_colour_dist_store = []
        target_zone_area_store = []

        #loop through adjacent zones
        for i in adjacent_regions:
          target_region_index = i
          #print target_region_index
          target_zone_colour_index = ([x for x, y in enumerate(region_colour_store) if y[0] == target_region_index])[0]
          target_zone_colour = region_colour_store[target_zone_colour_index]
          truth_mask_target = localise_region(mask_sum,target_region_index)
          target_zone_area = (np.nonzero(truth_mask_target)[0].shape)[0]
          target_zone_area_store.append([target_region_index,target_zone_area])
          region_colour_dist = calc_colour_dist(current_zone_colour,target_zone_colour)
          region_colour_dist_store.append([target_region_index,region_colour_dist,target_zone_colour_index])

        # outside loop
        region_colour_dist_store = np.asarray(region_colour_dist_store)
        target_zone_area_store = np.asarray(target_zone_area_store)
        min_index = np.argmin(region_colour_dist_store[:,1])
        min_region = region_colour_dist_store[min_index][0]
        min_region_colour_index = region_colour_dist_store[min_index][2]
        min_region_colour_index = np.int32(min_region_colour_index)
        min_region = np.int32(min_region)
        min_region_dist = region_colour_dist_store[min_index][1]
        min_region_area = target_zone_area_store[min_index][1]

        # if regions close enough, merge - else if A is small enough merge with min_dist region
        if min_region_dist < variable_list['euclid_region_thresh'] or truth_mask_contour_area < thresh4_area:
          region_colour_store[min_region_colour_index][1] = mean_region_colour(truth_mask_contour_area,min_region_area,current_zone_colour,min_region_colour_index,region_colour_store)
          mask_sum = merge_regions(current_region_index,min_region,truth_mask,mask_sum)

        region_store = np.unique(mask_sum)
        truth_mask_contours, hierarchy = cv2.findContours(hold_truth_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        number_of_regions_end = len(region_store)

    if number_of_regions_start == number_of_regions_end:
      thresh4_area = variable_list['region_growth_factor']*thresh4_area

  return mask_sum

#-------------------------------------------------------------------------------------------------

def convert_colourspace(img, colour_space):

  if colour_space == 'LAB':
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  elif colour_space == 'YCbCr':
   img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
  elif colour_space == 'BGR' or colour_space == 'RGB':
    pass
  else:
    print 'invalid colour space'

  return img

#-------------------------------------------------------------------------------------------------

def threshold_and_find_contours(img2,std_dev_img):

    retval,i = cv2.threshold(std_dev_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #calculate otsu threshold value 'retval'
    ret,th1 = cv2.threshold(std_dev_img,retval/variable_list['otsu_factor'],255,cv2.THRESH_BINARY_INV) #global threshold the image using 'retval' over otsu factor
    cv2.imwrite('binary.jpg',th1)
    contours, hierarchy = cv2.findContours(th1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    shape_one, shape_two, shape_three = hierarchy.shape[:]
    hierarchy = hierarchy.reshape(shape_two,shape_three)
    cv2.drawContours(img2, contours, -1, (0,255,0), 1)

    return contours, hierarchy

#-------------------------------------------------------------------------------------------------

def create_std_dev_img(img):

    blur1 = cv2.multiply(cv2.blur(img[:,:,0],(3,3)),cv2.blur(img[:,:,0],(3,3)),None,1,cv2.CV_64F)
    blur3 = cv2.multiply(cv2.blur(img[:,:,1],(3,3)),cv2.blur(img[:,:,1],(3,3)),None,1,cv2.CV_64F)
    blur5 = cv2.multiply(cv2.blur(img[:,:,2],(3,3)),cv2.blur(img[:,:,2],(3,3)),None,1,cv2.CV_64F)

    blur2 = cv2.multiply(img[:,:,0],img[:,:,0],0,1,cv2.CV_64F)
    blur2 = cv2.blur(blur2,(3,3))
    blur4 = cv2.multiply(img[:,:,1],img[:,:,1],0,1,cv2.CV_64F)
    blur4 = cv2.blur(blur4,(3,3))
    blur6 = cv2.multiply(img[:,:,2],img[:,:,2],0,1,cv2.CV_64F)
    blur6 = cv2.blur(blur6,(3,3))

    blur_output = cv2.sqrt(cv2.absdiff(blur2,blur1))+cv2.sqrt(cv2.absdiff(blur4,blur3))+cv2.sqrt(cv2.absdiff(blur6,blur5))
    blur_output = np.uint8(blur_output)

    return blur_output
#-------------------------------------------------------------------------------------------------

def collate_contours(mask_sum,img3):

  region_indicies = np.unique(mask_sum)
  contour_store = []

  for i in region_indicies:
    truth_mask = localise_region(mask_sum,i)
    contours_single, hierarchy = cv2.findContours(truth_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contour_single = contours_single[0]
    contour_store.append(contour_single)
    stacked_truth_masks = np.dstack((truth_mask,truth_mask,truth_mask))
    cv2.imshow('image',(stacked_truth_masks*img3))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  return contour_store


#-------------------------------------------------------------------------------------------------

def merge_regions(current_region_index,target_region_index,truth_mask,mask_sum):

  scaled_truth_mask = []
  scaled_truth_mask = np.int32(scaled_truth_mask)
  truth_mask = np.int32(truth_mask)
  scaled_truth_mask = truth_mask*target_region_index-(current_region_index*truth_mask)
  mask_sum += scaled_truth_mask

  return mask_sum

#-------------------------------------------------------------------------------------------------

def mean_region_colour(truth_mask_contour_area,min_region_area,current_zone_colour,min_region_colour_index,region_colour_store):

  current_zone_colour = np.array(current_zone_colour)
  target_colour = region_colour_store[min_region_colour_index][1]
  target_colour = np.array(target_colour)
  numerator = truth_mask_contour_area*current_zone_colour + min_region_area*(target_colour)
  denominator = truth_mask_contour_area + min_region_area
  mean_colour = numerator/denominator

  return mean_colour

#-------------------------------------------------------------------------------------------------

def find_adjacent_regions(mask_sum,truth_mask):

  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
  dilation = cv2.dilate(truth_mask,kernel,iterations = 1)
  band_mask = dilation-truth_mask
  band = band_mask*mask_sum
  region_indicies = band[np.nonzero(band)]
  region_indicies = np.unique(region_indicies)

  return region_indicies

#-------------------------------------------------------------------------------------------------

def localise_region(mask_sum,current_region_index):

  region_index_mask = np.ones(mask_sum.shape,np.int32)
  region_index_mask = region_index_mask*current_region_index
  truth_mask = (region_index_mask == mask_sum)
  truth_mask = np.uint8(truth_mask)

  return truth_mask

#-------------------------------------------------------------------------------------------------

def Add_T_element_to_region(T,mask_sum):

  Tprime = []
  x,y = T[0][0], T[0][1]
  zone = T[0][2]
  mask_sum[x,y,0] = zone

  i = T.pop(0)
  mask_sum_holder = mask_sum[:,:,0]
  mask_sum_bordered = cv2.copyMakeBorder(mask_sum_holder,1,1,1,1,cv2.BORDER_CONSTANT,value=1)

  i[0] += 1
  i[1] += 1

  pixel_up = [i[0]-1,i[1]]
  pixel_down = [i[0]+1,i[1]]
  pixel_left = [i[0],i[1]-1]
  pixel_right = [i[0],i[1]+1]

  if mask_sum_bordered[pixel_up[0],pixel_up[1]] == 0:
    Tprime.append([pixel_up[0]-1,pixel_up[1]-1])

  if mask_sum_bordered[pixel_down[0],pixel_down[1]] == 0:
    Tprime.append([pixel_down[0]-1,pixel_down[1]-1])

  if mask_sum_bordered[pixel_left[0],pixel_left[1]] == 0:
    Tprime.append([pixel_left[0]-1,pixel_left[1]-1])

  if mask_sum_bordered[pixel_right[0],pixel_right[1]] == 0:
    Tprime.append([pixel_right[0]-1,pixel_right[1]-1])

  return T,Tprime, mask_sum

#-------------------------------------------------------------------------------------------------

def initialise_Tprime(mask_sum):

  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
  mask_sum = mask_sum[:,:,0]
  mask_sum = np.uint8(mask_sum)
  dilation = cv2.dilate(mask_sum,kernel,iterations = 1)
  fin = dilation-mask_sum
  pixelpoints = np.transpose(np.nonzero(fin))

  return pixelpoints

#-------------------------------------------------------------------------------------------------

def Tprime_to_T(T, Tprime,img,mask_sum, region_colour_store):

  mask_sum = mask_sum[:,:,0]
  mask_sum_bordered = cv2.copyMakeBorder(mask_sum,1,1,1,1,cv2.BORDER_CONSTANT,value=0)

  for i in (Tprime):

    current_colour = img[i[0],i[1]]
    i[0] += 1
    i[1] += 1
    pixel_up = [i[0]-1,i[1]]
    pixel_down = [i[0]+1,i[1]]
    pixel_left = [i[0],i[1]-1]
    pixel_right = [i[0],i[1]+1]

    up_dist,down_dist,left_dist,right_dist = 1000,1000,1000,1000
    up_zone,down_zone,left_zone,right_zone = 0,0,0,0

    if mask_sum_bordered[pixel_up[0],pixel_up[1]] != 0:
      up_zone = mask_sum_bordered[pixel_up[0],pixel_up[1]]
      zone_colour_index = ([x for x, y in enumerate(region_colour_store) if y[0] == up_zone])[0]
      target_pixel_colour_up = region_colour_store[zone_colour_index]
      up_dist = calc_colour_dist(current_colour,target_pixel_colour_up)

    if mask_sum_bordered[pixel_down[0],pixel_down[1]] != 0:
      down_zone = mask_sum_bordered[pixel_down[0],pixel_down[1]]
      zone_colour_index = ([x for x, y in enumerate(region_colour_store) if y[0] == down_zone])[0]
      target_pixel_colour_down = region_colour_store[zone_colour_index]
      down_dist = calc_colour_dist(current_colour,target_pixel_colour_down)

    if mask_sum_bordered[pixel_left[0],pixel_left[1]] != 0:
      left_zone = mask_sum_bordered[pixel_left[0],pixel_left[1]]
      zone_colour_index = ([x for x, y in enumerate(region_colour_store) if y[0] == left_zone])[0]
      target_pixel_colour_left = region_colour_store[ zone_colour_index ]
      left_dist = calc_colour_dist(current_colour,target_pixel_colour_left)

    if mask_sum_bordered[pixel_right[0],pixel_right[1]] != 0:
      right_zone = mask_sum_bordered[pixel_right[0],pixel_right[1]]
      zone_colour_index = ([x for x, y in enumerate(region_colour_store) if y[0] == right_zone])[0]
      target_pixel_colour_right = region_colour_store[ zone_colour_index ]
      right_dist = calc_colour_dist(current_colour,target_pixel_colour_right)

    zones = [up_zone,down_zone,left_zone,right_zone]
    target_pixel_dists = [up_dist,down_dist,left_dist,right_dist]
    min_dist = np.amin(target_pixel_dists)

    if min_dist <1000:

      min_pixel = np.argmin(target_pixel_dists)
      T_converted = [i[0]-1,i[1]-1,zones[min_pixel],min_dist]

      if len(Tprime)>4:

        T.append(T_converted)

      else:

        x_values = []
        y_values = []
        duplicate_index = [x for x, y in enumerate(T) if y[0] == T_converted[0] and y[1] == T_converted[1]]

        if len(duplicate_index)>0: #T_converted[0] in x_values and T_converted[1] in y_values:

          index = duplicate_index[0]
          T.pop(index)

        keys = [r[3] for r in T]
        index_for_insertion = bisect.bisect_left(keys, T_converted[3])# doesn't seem to sort correctly so added next line
        T.insert(index_for_insertion,T_converted)

  Tprime = []

  return T,Tprime

#-------------------------------------------------------------------------------------------------

''' Reference: Shih page 3: equation (6), concerning the equation for Euclidean distance'''
def calc_colour_dist(current_colour,target_pixel_colour):
  target_pixel_colour = target_pixel_colour[1]

  zeroth_plane_dist = (current_colour[0] - target_pixel_colour[0])**2
  first_plane_dist = (current_colour[1] - target_pixel_colour[1])**2
  second_plane_dist = (current_colour[2] - target_pixel_colour[2])**2
  numerator = math.sqrt(zeroth_plane_dist+first_plane_dist+second_plane_dist)
  denominator = math.sqrt(current_colour[0]**2 + current_colour[1]**2 + current_colour[2]**2)
  colour_dist = numerator/(denominator+variable_list['euclid_bias']) # unknown what effect the 20 has here other than prevent crash
  return colour_dist

#-------------------------------------------------------------------------------------------------

def label_seed_regions(img,contours,hierarchy,region_colour_store):
  mask = np.zeros(img.shape,np.int32)
  mask_sum = np.zeros(img.shape,np.int32)
  I = 0

  for i in range(len(contours)):

    mask[:] = 0
    first_child = hierarchy[i][2]
    parent = hierarchy[i][3]

    if parent == -1:
      cv2.drawContours(mask,contours,i,(i+2),-1)

      if first_child != -1:

        cv2.drawContours(mask,contours,first_child,0,-1)
        next_child = first_child
        previous_child = first_child

        while hierarchy[next_child][0] != -1:

          cv2.drawContours(mask,contours,hierarchy[next_child][0],0,-1)
          next_child = hierarchy[next_child][0]

        while hierarchy[previous_child][1] != -1:

          cv2.drawContours(mask,contours,hierarchy[previous_child][1],0,-1)
          next_child = hierarchy[hierarchy[previous_child][1]][1]

      mask_holder = np.uint8(mask[:,:,0])
      mean_colour = cv2.mean(img,mask_holder)[0],cv2.mean(img,mask_holder)[1],cv2.mean(img,mask_holder)[2]
      region_colour_store.append([i+2,mean_colour])
      mask_sum += mask

    I += 1

  pixelpoints = np.transpose(np.nonzero(mask))

  return mask_sum, region_colour_store

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------


#run
Main(variable_list)


#-------------------------------------------------------------------------------------------------
