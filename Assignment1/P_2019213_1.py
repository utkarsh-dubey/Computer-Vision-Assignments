import numpy as np
import cv2
from math import *

#question 1 equation 3

def distance(x,y):
    return np.sqrt(np.sum((x-y)**2))


image = cv2.imread('leaf.png')
cv2.imshow('Input image',image)
pixel_vals = image.reshape((-1,3))
pixel_vals = np.float32(pixel_vals)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
retval, labels, centers = cv2.kmeans(pixel_vals, 85, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.array(centers,dtype='float64')
segmented_data = centers[labels.flatten()]
kmeanImage = segmented_data.reshape((image.shape))
print(len(centers))
kmeanImage = np.array(kmeanImage)
hashy = {}
n,m,o = kmeanImage.shape

# print(kmeanImage)
# for i in range(n):
#     for j in range(m):
#         if((kmeanImage[i][j][0]+kmeanImage[i][j][1]+kmeanImage[i][j][2])//3 not in hashy):
#             hashy[(kmeanImage[i][j][0]+kmeanImage[i][j][1]+kmeanImage[i][j][2])//3]=1
#         else:
#             hashy[(kmeanImage[i][j][0]+kmeanImage[i][j][1]+kmeanImage[i][j][2])//3]+=1

saliencyValues=[]
colorProb={}
total=0
for i in range(85):
    colorProb[i]=np.sum(labels.flatten()==i)
    total+=colorProb[i]
for i in colorProb:
    colorProb[i]=colorProb[i]/total

for i in colorProb:
    temp=0
    # print(i)
    for j in colorProb:
        # print(i,j)
        temp+=colorProb[j]*(distance(centers[i],centers[j]))
    saliencyValues.append(int(temp))

newSaliencyVAlues=[]
for i in labels.flatten():
    newSaliencyVAlues.append(saliencyValues[i])

outputImage = np.array(newSaliencyVAlues).reshape((n,m))

outputImage = 255*(outputImage-np.min(outputImage))/(np.max(outputImage)-np.min(outputImage))
outputImage = outputImage.astype(dtype='uint8')

cv2.imshow("Equation 3 output image",outputImage)
cv2.imwrite("equation3_output.png",outputImage)

cv2.waitKey()
cv2.destroyAllWindows()



#question 1 equation 5

image = cv2.imread('segmented_big_tree.jpg')
image = np.array(image)
# n,m,_=image.shape

# image = cv2.imread('BigTree')
# cv2.imshow('Input image',image)
pixel_vals = image.reshape((-1,3))
pixel_vals = np.float32(pixel_vals)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
retval, labels, centers = cv2.kmeans(pixel_vals, 15, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.array(centers,dtype='float64')
segmented_data = centers[labels.flatten()]
kmeanImage = segmented_data.reshape((image.shape))
print(len(centers))
kmeanImage = np.array(kmeanImage)
hashy = {}
n,m,o = kmeanImage.shape

# print(kmeanImage)
# for i in range(n):
#     for j in range(m):
#         if((kmeanImage[i][j][0]+kmeanImage[i][j][1]+kmeanImage[i][j][2])//3 not in hashy):
#             hashy[(kmeanImage[i][j][0]+kmeanImage[i][j][1]+kmeanImage[i][j][2])//3]=1
#         else:
#             hashy[(kmeanImage[i][j][0]+kmeanImage[i][j][1]+kmeanImage[i][j][2])//3]+=1

saliencyValues=[]
colorProb={}
total=0
for i in range(15):
    colorProb[i]=np.sum(labels.flatten()==i)
    total+=colorProb[i]
for i in colorProb:
    colorProb[i]=colorProb[i]/total

for i in colorProb:
    temp=0
    # print(i)
    for j in colorProb:
        # print(i,j)
        temp+=colorProb[j]*(distance(centers[i],centers[j]))
    saliencyValues.append(int(temp))

newSaliencyVAlues=[]
for i in labels.flatten():
    newSaliencyVAlues.append(saliencyValues[i])

outputImage = np.array(newSaliencyVAlues).reshape((n,m))

outputImage = 255*(outputImage-np.min(outputImage))/(np.max(outputImage)-np.min(outputImage))
outputImage = outputImage.astype(dtype='uint8')

cv2.imshow("Equation 5 output image",outputImage)
cv2.imwrite("equation5_output.png",outputImage)
cv2.waitKey()








