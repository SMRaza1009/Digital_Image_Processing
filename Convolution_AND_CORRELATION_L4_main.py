import cv2
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

# ------------- MEAN MEDIAN MODE ---------------------- #

#flower_Img= cv2.imread('E:/Research/Digital Image Processing Python/flower_2.jpg')
flower_Img= cv2.imread('E:/Research/Digital Image Processing Python/flowerSalt.jpg')
flower_Img = cv2.resize(flower_Img,(800,600)) # Resize image

# ----------- MEAN CONDITION -------------- #
#flt = np.array([(1,1,1,1,1),(1,1,1,1,1),(1,1,1,1,1),(1,1,1,1,1),(1,1,1,1,1)])*(1/25) #filter (5,5)
#flt = np.array([(1,1,1),(1,1,1),(1,1,1)])*(1/9) # filter (3,3)

#------------------ MEDIAN CONDITION---------#
flt = np.array([(0,0,0),(0,0,0),(0,0,0)])

s= flower_Img.shape
F= flt.shape

flowerGray = cv2.cvtColor(flower_Img,cv2.COLOR_BGR2GRAY) # Binary conversion to grayscale image
cv2.imshow("Original-GrayScale", flowerGray)


#---------- APPPLYING ZERO PADDING---------#
R = s[0]+F[0]-1
C = s[1]+F[1]-1
Z = np.zeros((R,C))

for i in range(s[0]):
    for j in range(s[1]):
        Z[i+np.int((F[0]-1)/2),j+np.int((F[1]-1)/2)] = flowerGray[i,j]
        
print(Z)        

#----------  Applying Filter on Padding Image -------- #
for i in range(s[0]):
    for j in range(s[1]):
        k = Z[i:i+F[0], j:j+F[1]] # extract values when appling filter
#       l = np.sum(k*flt) # MEAN FILTER
        #l = np.median(k) # Median Filter
        l = stats.mode(k, axis=None) # Mode Filter
        #flowerGray[i,j] = l
        flowerGray[i,j]= l[0][0]
#cv2.imshow('Final Output Image 5 x 5 MEAN FILTER ', flowerGray) #-- MEAN FILTER
#cv2.imshow('MEDIAN FILTER ', flowerGray)
cv2.imshow('MODE FILTER ', flowerGray)
cv2.waitKey(0)        