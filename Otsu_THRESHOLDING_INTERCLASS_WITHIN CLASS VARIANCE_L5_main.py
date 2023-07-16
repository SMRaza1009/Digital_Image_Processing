import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import math


image = cv2.imread('E:/Research/Digital Image Processing Python/flower.jpg')
image = cv2.resize(image,(800, 600))

imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('Original GrayScale Image',imageGray)
#cv2.waitKey(0)

# Histogram

hist = cv2.calcHist([imageGray],[0], None,[255], [0,255])
print(hist)

within = []
between = []
d = 0
# We will finding weight, mean, variance in the form of x, x1, x2, x3 and y, y1, y2, y3
for i in range(len(hist)):
    x, y = np.split(hist,[i])
    x1 = np.sum(x)/(image.shape[0]*image.shape[1]) #weight of the class
    y1 = np.sum(y)/(image.shape[0]*image.shape[1])
    x2 = np.sum([ j*t for j,t in enumerate(x)])/np.sum(x) # Mean
    x2 = np.nan_to_num(x2)
    y2 = np.sum([ (j+d)*(t) for j,t in enumerate(y)])/np.sum(y)
    x3 = np.sum([(j-x2)**2*t for j,t in enumerate(x)])/np.sum(x)
    x3 = np.nan_to_num(x3)
    y3 = np.sum([(j+d-y2)**2*t for j,t in enumerate(y)])/np.sum(y)
    d = d+1
    between.append(x1*y1*(x2-y2)*(x2-y2))
    within.append(x1*x3 + y1*y3) # within class variance
m = np.argmin(within) # minimum value for variance
n = np.argmax(between) # maximum value for variance
print(m)    
print (n)
(thresh, Bin) = cv2.threshold(imageGray,m,255,cv2.THRESH_BINARY)
cv2.imshow("Binarization", Bin)
cv2.waitKey(0)