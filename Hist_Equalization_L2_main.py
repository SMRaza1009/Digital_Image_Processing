import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

flower = cv2.imread('E:/Research/Digital Image Processing Python/flowers/bellflower/1.jpg')
s = flower.shape
#print(s)
#cv2.imshow('Original',flower)
flowerGray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
flowerGray =  cv2.convertScaleAbs(flowerGray, alpha=1.10, beta= -50)
#cv2.imshow('Binary N GrayScale converted image',flowerGray)
#cv2.waitKey(0)


# ----------- Histogram Equalization -----------#
def Hist(image):
    s = image.shape
    histEq = np.zeros(shape=(256,1))
    for i in range(s[0]):
        for j in range(s[1]):
            k = image[i, j]
            histEq[k,0]= histEq[k,0]+1
    return histEq


histg = Hist(flowerGray)
plt.plot(histg)
x= histg.reshape(1,256)
y = np.array([])
y = np.append(y,x[0,0])

for i in range(255):
    k = x[0,i+1]+y[i]
    y = np.append(y,k)
y = np.round((y/(s[0]*s[1]))*(256-1))
 
for i in range(s[0]):
    for j in range(s[1]):
        k = flowerGray[i,j]
        flowerGray[i,j] = y[k]    
        
equal = Hist(flowerGray)
plt.title("Histogram Equalization")
plt.plot(equal)
plt.show()   
cv2.waitKey(0)    
#cv2.imshow("Histogram Equalization",flowerGray)            
