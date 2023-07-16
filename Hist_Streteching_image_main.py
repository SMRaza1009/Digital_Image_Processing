import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

flower = cv2.imread('E:/Research/Digital Image Processing Python/flowers/black_eyed_susan/1.jpg')
flower = cv2.resize(flower,(224,224))
s = flower.shape
flowerGray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
flowerGray = cv2.convertScaleAbs(flowerGray, alpha=1.10, beta=40)
cv2.imshow('Original',flowerGray)
#cv2.waitKey(0)


# -------------Histogram Stretching-------------#

# Histogram Function
def Hist(image):
    s = image.shape
    H = np.zeros(shape=(256,1))
    for i in range(s[0]):
        for j in range(s[1]):
            k= image[i,j]
            H[k,0] = H[k,0]+1
    return H

histg = Hist(flowerGray)
#plt.title('Histogram of an image')
#plt.plot(histg)
#plt.show()

# ------------Main Code of Stretch--------------#

x = histg.reshape(1,256)
y = np.zeros((1,256))

for i in range(256):
    if x[0,i] == 0:
        y[0,i]=0
    else:
        y[0,i]=i

min = np.min(y[np.nonzero(y)])
max = np.max(y[np.nonzero(y)])

strech = np.round(((255-0)/(max-min)*(y-min)))
strech[strech<0] = 0
strech[strech>0]=255

for i in range(s[0]):
    for j in range(s[1]):
        k = flowerGray[i,j]
        flowerGray[i,j]= strech[0,k]

histg2 = Hist(flowerGray)
cv2.imshow('Stretching Image', flowerGray)
plt.title('Stretching Image')
plt.plot(histg)
plt.plot(histg2)
plt.show()                     
    
        