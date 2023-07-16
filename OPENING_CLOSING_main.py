import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os

#image = cv2.imread('E:/Research/Digital Image Processing Python/Morph2.jpg') # for Erosion Method
image = cv2.imread('E:/Research/Digital Image Processing Python/Morph3.jpg') # For Dilation Method
cv2.imshow('Org', image)
imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(thresh, binary) = cv2.threshold(imgGray,172,255,cv2.THRESH_BINARY)
#cv2.imshow('Binary', bin)

filt = np.array([(1,1,1,1,1,1,1),(1,1,1,1,1,1,1),(1,1,1,1,1,1,1),(1,1,1,1,1,1,1),(1,1,1,1,1,1,1),(1,1,1,1,1,1,1),(1,1,1,1,1,1,1)])

def erosion(img,filt):
    S = img.shape
    F = filt.shape
    img = img/255 # Converting 0 to 255 image values to 0 to 1 ,1
    R = S[0] + F[0] - 1 
    C = S[1] + F[1] - 1
    N = np.zeros((R,C)) # Created new array to store result values

    for i in range(S[0]):
        for j in range(S[1]):
            N[i+1, j+1] = img[i,j]
            #N[i+np.int((F[0]-1)/2), j+np.int((F[1]-1)/2)] # Padding the image
            
            


    for i in range(S[0]):  # 2D image value store for Erosion
        for j in range(S[1]):
            k = N[i:i+F[0], j:j+F[1]] # selecting window size through filter
            result = (k == filt)
            final = np.all(result == True) # Erosion Method
            #final = np.any(result == True) # Dilation Method
            if final:
                img[i,j] = 1
            else:
                img[i,j] = 0
    return img*255

def dilation(img,filt):
    S = img.shape
    F = filt.shape
    img = img/255 # Converting 0 to 255 image values to 0 to 1 ,1
    R = S[0] + F[0] - 1 
    C = S[1] + F[1] - 1
    N = np.zeros((R,C)) # Created new array to store result values

    for i in range(S[0]):
        for j in range(S[1]):
            N[i+1, j+1] = img[i,j]
            #N[i+np.int((F[0]-1)/2), j+np.int((F[1]-1)/2)] # Padding the image
            
            


    for i in range(S[0]):  # 2D image value store for Erosion
        for j in range(S[1]):
            k = N[i:i+F[0], j:j+F[1]] # selecting window size through filter
            result = (k == filt)
            final = np.any(result == True) # Erosion Method
            #final = np.any(result == True) # Dilation Method
            if final:
                img[i,j] = 1
            else:
                img[i,j] = 0
    return img*255

def opeining(img,filt):
    opening1 = erosion(binary,filt)
    opening2 = dilation(opening1,filt)
    return opening2

def closing(img,filt):
    closing1 = dilation(binary,filt)
    closing2 = erosion(closing1,filt)
    return closing2
     
final = closing(binary,filt)     
#cv2.imshow('Final-Opening',opening2)
cv2.imshow('Final-Opening',final)
cv2.waitKey(0)                
        
