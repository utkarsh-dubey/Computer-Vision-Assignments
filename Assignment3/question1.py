from turtle import color
import cv2
from math import *
import numpy as np
from skimage.segmentation import slic

image = np.array(cv2.imread('0002.jpg'),dtype='float')
# print(image.shape)
image/=255.0   #converts image pixel values from 0-1
def square(x,y):
    return (x-y)**2
def formulaCalculation(x,y,i,j):
    global image
    return ((square(x[i][0],x[j][0])+square(x[i][1],x[j][1])+square(x[i][2],x[j][2]))**(0.5))*(np.exp((square(y[i][0],y[j][0])+square(y[i][1],y[j][1]))**(0.5)/(square(image.shape[0],image.shape[1]))**(0.5)))


# image = np.array(cv2.imread('0002.jpg'),dtype='float')
# # print(image.shape)
# image/=255.0   #converts image pixel values from 0-1

cv2.imshow("input image",image)
cv2.waitKey()

numOfSegments = [50,100,150,200,250]
for num in numOfSegments:
    segments = slic(image, n_segments = num, sigma = 5)
    superPixels = np.unique(segments)
    representativeColors = np.zeros((superPixels.shape[0],3))
    representativePixels = np.zeros((superPixels.shape[0],2))

    for i in range(superPixels.shape[0]):

        pixels = np.where(segments==superPixels[i])
        colors = np.array([image[pixels[0][j]][pixels[1][j]] for j in range(len(pixels[0]))])
        # print(colors.shape)
        r,b,g,x,y = 0,0,0,0,0

        for j in range(colors.shape[0]):
            b+=(colors[j][0]/colors.shape[0])
            g+=(colors[j][1]/colors.shape[0])
            r+=(colors[j][2]/colors.shape[0])
            x+=(pixels[0][j]/colors.shape[0])
            y+=(pixels[1][j]/colors.shape[0])
        representativeColors[i][0]=b
        representativeColors[i][1]=g
        representativeColors[i][2]=r
        representativePixels[i][0]=int(x)
        representativePixels[i][1]=int(y)

    representativeColors = (representativeColors*255).astype(int)
    allSaliency=[]
    for i in range(superPixels.shape[0]):
        saliency=0
        for j in range(superPixels.shape[0]):
            saliency+=formulaCalculation(representativeColors,representativePixels,i,j)
        allSaliency.append(saliency)
    
    maxi = max(allSaliency)
    for i in range(len(allSaliency)):
        allSaliency[i]/=maxi
        allSaliency[i]*=255
    
    saliencyMap = np.zeros((image.shape[0],image.shape[1]),np.uint8)

    for i in range(superPixels.shape[0]):
        pixels = np.where(segments==superPixels[i])
        for j in range(len(pixels[0])):
            saliencyMap[pixels[0][j]][pixels[1][j]] = allSaliency[i]
    
    cv2.imshow("Saliency Map with segments = "+str(num),saliencyMap)
    cv2.imwrite("./Saliency Map with segments = "+str(num)+".jpg",saliencyMap)
cv2.waitKey()


