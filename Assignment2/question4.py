import cv2
import glob
import numpy as np
from math import *

kMain=5
noOfCorners=10
inputImage = cv2.imread('A:\Semester 6\CV\Assignment2\cat.jpg',0)
cv2.imshow("input",inputImage)
corners = cv2.goodFeaturesToTrack(inputImage,noOfCorners,0.1,10)
corners = np.int0(corners)
patchVector = []

cv2.waitKey()
for i in range(noOfCorners):
    vector = np.zeros((7,7))
    k=0
    for j in range(corners[i][0][1]-3,corners[i][0][1]+3):
        for k in range(corners[i][0][0]-3,corners[i][0][0]+3):
            try:
                p7=(inputImage[j][k] > inputImage[j-1][k-1])
                p6=(inputImage[j][k] > inputImage[j-1][k])
                p5=(inputImage[j][k] > inputImage[j-1][k+1])
                p4=(inputImage[j][k] > inputImage[j][k+1])
                p3=(inputImage[j][k] > inputImage[j+1][k+1])
                p2=(inputImage[j][k] > inputImage[j+1][k])
                p1=(inputImage[j][k] > inputImage[j+1][k-1])
                p0=(inputImage[j][k] > inputImage[j][k-1])
                vector[corners[i][0][1]-j-3][corners[i][0][0]-k-3]=p7*2**7 + p6*2**6 + p5*2**5 + p4*2**4 + p3*2**3 + p2*2**2 + p1*2**1 + p0*2**0
            except:
                continue
    patchVector.append(vector)
# print(vector)

featureVector = []
for i in patchVector:
    mean = np.mean(i)
    std = np.std(i)
    featureVector.append(mean)
    featureVector.append(std)

# closest = np.zeros((k,2),np.uint8)
closest = []

for i in range(kMain):
    closest.append([0,255**2*8])

images = [cv2.imread(file,0) for file in glob.glob('dd/*.jpg')]

for image in range(len(images)):
    imageCorners = cv2.goodFeaturesToTrack(images[image],noOfCorners,0.1,10)
    imageCorners = np.int0(imageCorners)

    # for i in range(noOfCorners):
    #     if(imageCorners[i][0][1]==1):
    #         imageCorners[i][0][1]+=1
    #     if(imageCorners[i][0][1]==len(images[image][0])):
    #         imageCorners[i][0][1]-=1
    #     if(imageCorners[i][0][0]==1):
    #         imageCorners[i][0][0]+=1
    #     if(imageCorners[i][0][0]==len(images[image])):
    #         imageCorners[i][0][0]-=1
    
    tempPatchVector = []

    k=0
    for i in range(noOfCorners):
        tempVector = np.zeros((7, 7))
        for j in range(imageCorners[i][0][1]-3,imageCorners[i][0][1]+3):
            for k in range(imageCorners[i][0][0]-3,imageCorners[i][0][0]+3):
                try:
                    p7=(images[image][j][k] > images[image][j-1][k-1])
                    p6=(images[image][j][k] > images[image][j-1][k])
                    p5=(images[image][j][k] > images[image][j-1][k+1])
                    p4=(images[image][j][k] > images[image][j][k+1])
                    p3=(images[image][j][k] > images[image][j+1][k+1])
                    p2=(images[image][j][k] > images[image][j+1][k])
                    p1=(images[image][j][k] > images[image][j+1][k-1])
                    p0=(images[image][j][k] > images[image][j][k-1])
                    tempVector[imageCorners[i][0][1]-j-3][imageCorners[i][0][0]-k-3]=p7*2**7 + p6*2**6 + p5*2**5 + p4*2**4 + p3*2**3 + p2*2**2 + p1*2**1 + p0*2**0
                except:
                    continue
        tempPatchVector.append(tempVector)
    tempFeatureVector = []
    for i in tempPatchVector:
        mean = np.mean(i)
        std = np.std(i)
        tempFeatureVector.append(mean)
        tempFeatureVector.append(std)
    tempSum = np.linalg.norm(np.subtract(tempFeatureVector, featureVector))
    for i in range(kMain):
        if(closest[i][1]>tempSum):
            for j in range(kMain-1,i,-1):
                # print("j",j)
                closest[j][0]=closest[j-1][0]
                closest[j][1]=closest[j-1][1]
            # closest[i+1:] = closest[i:-1]
            closest[i][0]=image
            closest[i][1]=tempSum
            break
for i in range(kMain):
    print(closest[i][1])
    cv2.imshow("output"+str(i),images[closest[i][0]])
    cv2.imwrite("nearest"+str(i+1)+".jpg",images[closest[i][0]])
    
cv2.waitKey()



