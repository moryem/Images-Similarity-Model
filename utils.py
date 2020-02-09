"""
Created on Tue Oct 23 13:11:45 2018

@author: Mor
"""

import _pickle as pickle
import numpy as np
import os
import cv2
from sklearn.utils import shuffle 
import matplotlib.pyplot as plt


def load_CIFAR_batch(filename):
# =============================================================================
#     load single batch of cifar
# =============================================================================

    with open(filename, 'rb') as f:
        datadict = pickle.load(f,encoding='latin1')
        X = datadict['data']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1)
        X = np.divide(X, 255) # Normalize
    
    return X

def load_CIFAR10(ROOT):
# =============================================================================
#     load all of cifar
# =============================================================================

    # create the train set
    xs = []

    for b in range(1,5):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X = load_CIFAR_batch(f)
        xs.append(X)
    Xtr = np.concatenate(xs)

    del X
    
    # load validation set
    Xval = load_CIFAR_batch(os.path.join(ROOT, 'data_batch_5'))
    
    # load test set
    Xte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  
    return Xtr, Xval, Xte

def circle_rotate(image, x, y, radius):
# =============================================================================
#   circle rotate a patch of the image
# =============================================================================
    
    crop = image[x-radius:x+radius+1,y-radius:y+radius+1,:]
    
    # build the cirle mask
    mask = np.zeros(crop.shape) #(2*radius+1, 2*radius+1,crop.shape[2])
    for i in range(crop.shape[0]):
        for j in range(crop.shape[1]):
            if (i-radius)**2 + (j-radius)**2 <= radius**2:
                mask[i,j,:] = 1
                
    # create the new circular image
    sub_img = np.empty(crop.shape ,dtype='uint8')
    sub_img = mask * crop  
    angle = np.random.randint(40,125) # random angle between 40 to 125 degrees
    M = cv2.getRotationMatrix2D((crop.shape[0]/2,crop.shape[1]/2),angle,1)
    dst = cv2.warpAffine(sub_img,M,(crop.shape[0],crop.shape[1]))  

    # return the whole image after distortion
    i2 = image.copy()
    i2[x-radius:x+radius+1,y-radius:y+radius+1,:] = crop * (1-mask)
    i2[x-radius:x+radius+1,y-radius:y+radius+1,:] += dst
    
    return i2    

def translate(image, x, y, length):
# =============================================================================
#   translate a patch in the image    
# =============================================================================

    crop = image[x:x+length,y:y+length,:]
                
    # translate
    rand_x = int(np.random.randint(0,crop.shape[0]/2)) # random translation on x axis
    rand_y = int(np.random.randint(0,crop.shape[1]/2)) # random translation on y axis
    M = np.float32([[1,0,rand_x],[0,1,rand_y]])
    dst = cv2.warpAffine(crop,M,(crop.shape[0],crop.shape[1]))
    
    # create the mask
    mask = np.zeros(crop.shape,'int')
    ind_zero = np.where(dst==0)
    mask[ind_zero] = 1
    
    # return the whole image after distortion
    i2 = image.copy()
    i2[x:x+length,y:y+length,:] = crop * mask
    i2[x:x+length,y:y+length,:] += dst
    
    return i2

def affine(image, x, y, length):
# =============================================================================
#   affine transformation of a patch in the image    
# =============================================================================

    crop = image[x:x+length,y:y+length,:]
                
    # affine transformation
    s10 = (0,0)
    s11 = (crop.shape[1],0)
    s12 = (0,crop.shape[0])
    pts1 = np.float32([s10,s11,s12])
    s20 = (crop.shape[0] * 0.0,crop.shape[1] * 0.33)
    s21 = (crop.shape[0] * 0.85,crop.shape[1] * 0.25)
    s22 = (crop.shape[0] * 0.15,crop.shape[1] * 0.7)
    pts2 = np.float32([s20,s21,s22])
    M = cv2.getAffineTransform(pts1,pts2)    
    dst = cv2.warpAffine(crop,M,(crop.shape[0],crop.shape[1]))
    
    # create the mask
    mask = np.zeros(crop.shape,'int')
    ind_zero = np.where(dst==0)
    mask[ind_zero] = 1
    
    # return the whole image after distortion
    i2 = image.copy()
    i2[x:x+length,y:y+length,:] = crop * mask
    i2[x:x+length,y:y+length,:] += dst
    
    return i2

def brightness(image, x, y, length):
# =============================================================================
#   change the brightness of the image
# =============================================================================
    
    crop = image[x:x+length,y:y+length,:]
    
    # change brightness
    brightness = np.random.uniform(0.4,0.8)   
    factor = np.random.uniform(-brightness,brightness)
    dst = crop + factor

    # create the mask
    mask = np.zeros(crop.shape,'int')
    ind_zero = np.where(dst==0)
    mask[ind_zero] = 1
    
    # return the whole image after distortion
    i2 = image.copy()
    i2[x:x+length,y:y+length,:] = crop * mask
    i2[x:x+length,y:y+length,:] += dst
    
    return i2

def contrast(image, x, y, length):
# =============================================================================
#   change the contrast of the image
# =============================================================================
    
    crop = image[x:x+length,y:y+length,:]
    
    # change contrast
    contrast = np.random.uniform(0.4,0.7) 
    factor = np.random.uniform(contrast,contrast)
    image_mean = np.mean(np.mean(crop,axis=0),axis=0)
    dst = (crop - image_mean) * factor + image_mean

    # create the mask
    mask = np.zeros(crop.shape,'int')
    ind_zero = np.where(dst==0)
    mask[ind_zero] = 1
    
    # return the whole image after distortion
    i2 = image.copy()
    i2[x:x+length,y:y+length,:] = crop * mask
    i2[x:x+length,y:y+length,:] += dst

    return i2

def flip(image, x, y, length):
# =============================================================================
#     flip on x - axis
# =============================================================================

    crop = image[x:x+length,y:y+length,:]

    # flip on x axis
    dst = cv2.flip(crop, 1)
    
    # create the mask
    mask = np.zeros(crop.shape,'int')
    ind_zero = np.where(dst==0)
    mask[ind_zero] = 1
    
    # return the whole image after distortion
    i2 = image.copy()
    i2[x:x+length,y:y+length,:] = crop * mask
    i2[x:x+length,y:y+length,:] += dst

    return i2
   
def dist_part(img, perc, func):
# =============================================================================
#     send to one of the distortions
# =============================================================================

    if func.__name__ == 'circle_rotate':
        # radius length
        radius = int(np.round(np.sqrt(perc) * img.shape[0])/2-1)
        # center location
        rand_x = np.random.randint(low = radius, high = img.shape[0]-radius)
        rand_y = np.random.randint(low = radius, high = img.shape[1]-radius)
        # distort
        dst = func(img, rand_x, rand_y, radius)
    else:       
        # size of patch
        x = int(np.round(np.sqrt(perc) * img.shape[0]))    
        y = int(np.round(np.sqrt(perc) * img.shape[1]))
        
        # generate random locations
        rand_ind_x = np.random.randint(low = 0, high = img.shape[0]-x)
        rand_ind_y = np.random.randint(low = 0, high = img.shape[1]-y)
        
        # distort the patch
        dst = func(img, rand_ind_x, rand_ind_y, x)
    
    return dst
    
def matched_pair_perc(X):
# =============================================================================
#   create the parallel set for the siamese net and the labels
# =============================================================================
    
    # similar pairs
    Cs = X[:int(X.shape[0]/3),:,:,:]
    
    # totaly different pairs
    Cdf = Cs
    
    # level of distortion
    y_s = np.concatenate((np.zeros((Cs.shape[0],1),dtype=int),np.ones((Cdf.shape[0],1),dtype=int)))    

    # one distorted image in a pair
    dist_X =  X[int(2*X.shape[0]/3):,:,:,:]
    p = np.linspace(0,1,30)
    p = p[1:len(p)-1]
    funcs = [circle_rotate, translate, affine, brightness, contrast, flip] 
    Cds = []
    for x in dist_X:
        i = np.random.randint(0,len(p))
        j =  np.random.randint(0,len(funcs))
        ds_x = dist_part(x, p[i] ,funcs[j])
        Cds.append(ds_x)
        y_s = np.append(y_s,p[i])
    Cds = np.asarray(Cds)

    # total pairs
    C = np.concatenate((Cs,Cdf,Cds),axis=0)    

    # create the labels - 1 if totaly different and 0 otherwise
    y = np.concatenate((np.zeros((Cs.shape[0],1),dtype=int),np.ones((Cdf.shape[0],1),
                                 dtype=int),np.zeros((Cds.shape[0],1),dtype=int)))    
    # shuffle the data
    X, C, y, y_s = shuffle(X, C, y, y_s, random_state=52)
    
    return X, C, y, y_s

def data_prep_perc(ROOT):
# =============================================================================
#     prepare the data for train, validation and test
# =============================================================================

    Xtr, Xval, Xte = load_CIFAR10(ROOT)
    
    Xtr, Ctr, y_tr, l_tr = matched_pair_perc(Xtr)
    Xval, Cval, y_val, l_val = matched_pair_perc(Xval)
    Xte, Cte, y_te, l_te = matched_pair_perc(Xte)

    return Xtr, Xval, Xte, Ctr, Cval, Cte, y_tr, y_val, y_te, l_tr, l_val, l_te

def visualize(twin_net, img, ref, im_num):
# =============================================================================
#   Visualize the outputs of the twins net    
# =============================================================================
    
    # Input images
    img_tensor = np.expand_dims(img,axis=0)
    ref_tensor = np.expand_dims(ref,axis=0)
    
    # Right twin's output
    right_output = twin_net.predict(img_tensor)
    left_output = twin_net.predict(ref_tensor)
    
    # plot the output image and it's refernce image
    plt.subplot(2,2,1)
    plt.imshow(img); plt.axis('off')
    plt.title('Right twin input')
    plt.subplot(2,2,2)
    plt.imshow(ref); plt.axis('off')
    plt.title('Left twin input')
    plt.subplot(2,2,3)
    plt.imshow(np.squeeze(right_output, axis=0)); plt.axis('off')
    plt.title('Right twin output')
    plt.subplot(2,2,4)
    plt.imshow(np.squeeze(left_output, axis=0)); plt.axis('off')
    plt.title('Left twin output')
    plt.savefig('vis_examp ' + str(im_num))
    plt.show(); plt.close()

