import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os

flower = cv2.imread('E:/Research/Digital Image Processing Python/flowers/astilbe/1.jpg')
flower = cv2.resize(flower, (224,224))
#cv2.imshow('Original',flower)
#cv2.waitKey(0)
s = flower.shape # it will give the size and channel information
#print(s)

flowerGray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY) # Convert RGB to GRAY-SCALE
#cv2.imshow('Binary-image',flowerGray)
flowerGray = cv2.convertScaleAbs(flowerGray, alpha= 1.10, beta = -50) # Alpha is contrast and Beta is brighness     
#cv2.imshow('Enchance',flowerGray)
#cv2.waitKey(0)

Hist = np.zeros(shape=(256,1))
print(Hist)

for i in range(s[0]): # For rows-wise
    for j in range(s[1]): # For coloumn-wise
        k = flowerGray[i,j] # Read image value from using for loop and storing in K variable
        Hist[k,0] = Hist[k,0]+1
#print(Hist)
plt.title("Histogram of Image")
plt.plot(Hist)
plt.show()        



