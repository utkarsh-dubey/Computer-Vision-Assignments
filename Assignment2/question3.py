from unittest.mock import patch
import cv2
import glob
import numpy as np
from math import *
from skimage.feature import hog
from sklearn.cluster import KMeans

images = [cv2.imread(file) for file in glob.glob('dd/*.jpg')]
images = [cv2.resize(image,(256,256),interpolation=cv2.INTER_CUBIC) for image in images]

def doHog(images):
    hogFeatures = []

    for image in images:
        fd = hog(image, orientations=10, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False,multichannel=True)
        patch =[]

        for i in range(len(fd)):
            if(i==0):
                patch.append(fd[i])
                continue
            if(i%10==0):
                hogFeatures.append(patch)
                patch=[]
            patch.append(fd[i])
        hogFeatures.append(patch)
    
    return hogFeatures

hogFeatures = doHog(images)       
print(len(hogFeatures[0]))
k=4

kmeans = KMeans(n_clusters=k,random_state=0)
kmeans = kmeans.fit(hogFeatures)
print(len(kmeans.labels_))
print(kmeans.labels_)
print(kmeans.cluster_centers_)
centreCluster = list(kmeans.cluster_centers_)
inputImage = cv2.resize(cv2.imread("cat.jpg"), (256,256), interpolation=cv2.INTER_CUBIC)

cv2.imshow("input image",inputImage)
cv2.waitKey()

hogFeaturesInputImage = doHog([inputImage])

print("Total input image hog features ",len(hogFeaturesInputImage))

featureVector = np.zeros(k)

for i in range(len(hogFeaturesInputImage)):
    minDist = inf
    minPatchIndex = 0
    for j in range(len(centreCluster)):
        distance = np.linalg.norm(hogFeaturesInputImage[i] - centreCluster [j])
        if(distance<minDist):
            minPatchIndex = j
            minDist = distance
        
    featureVector[minPatchIndex]+=1
    
print("Feature vectors of input image", featureVector)
