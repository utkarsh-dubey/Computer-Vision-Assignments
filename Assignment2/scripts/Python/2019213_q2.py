import cv2
import glob
import numpy as np
from math import *
from fcmeans import FCM

images = [cv2.imread(file,0) for file in glob.glob('dd/*.jpg')]
images = [cv2.resize(image,(256,256),interpolation=cv2.INTER_CUBIC) for image in images]


def value(image,centre,x,y):
    m,n=image.shape
    if(x<0 or x>=m):
        return 0
    if(y<0 or y>=n):
        return 0
    if(max(image[x][y],centre)==0):
        return 0
    return round((min(image[x][y],centre))/(max(image[x][y],centre)))

def pixelValue(image,x,y):

    val=0
    val+=value(image,image[x][y],x,y-1)*1
    val+=value(image,image[x][y],x+1,y-1)*2
    val+=value(image,image[x][y],x+1,y)*2*2
    val+=value(image,image[x][y],x+1,y+1)*2*2*2
    val+=value(image,image[x][y],x,y+1)*2*2*2*2
    val+=value(image,image[x][y],x-1,y+1)*2*2*2*2*2
    val+=value(image,image[x][y],x-1,y)*2*2*2*2*2*2
    val+=value(image,image[x][y],x-1,y-1)*2*2*2*2*2*2*2

    return val
    

def lbp(image):
    lbpImage = np.zeros(image.shape,dtype='uint8')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            lbpImage[i][j] = pixelValue(image,i,j)
    
    return lbpImage


def spp(images):
    featureVectors = []
    for image in images:
        featureVector = []
        patch16=[image[x:x+64,y:y+64] for x in range(0,image.shape[0],64) for y in range(0,image.shape[1],64)]
        patch4=[image[x:x+128,y:y+128] for x in range(0,image.shape[0],128) for y in range(0,image.shape[1],128)]
        patch2=[image[x:x+256,y:y+128] for x in range(0,image.shape[0],256) for y in range(0,image.shape[1],128)]
        for i in range(len(patch16)):
            featureVector.append(np.mean(patch16[i]))
            featureVector.append(np.std(patch16[i]))
        for i in range(len(patch4)):
            featureVector.append(np.mean(patch4[i]))
            featureVector.append(np.std(patch4[i]))
        for i in range(len(patch2)):
            featureVector.append(np.mean(patch2[i]))
            featureVector.append(np.std(patch2[i]))
        featureVector.append(np.mean(image))
        featureVector.append(np.std(image))
        featureVectors.append(featureVector)
    
    return np.array(featureVectors)



lbpImages=[]

for image in images:
    lbpImages.append(lbp(image))

featureVectors = spp(lbpImages)
print(featureVectors.shape)
model = FCM(n_clusters=4)
model.fit(featureVectors)

centres = model.centers
labels = model.predict(featureVectors)
print(labels)

clusters = []
for i in range(4):
    clusters.append([])

for i in range(len(images)):
    clusterNumber = model.predict(featureVectors[i])[0]
    clusters[clusterNumber].append(images[i])
    final_path=".\\outputs\\question2\\cluster"+str(clusterNumber)+'\\'+str(i+1)+".jpg"
    cv2.imwrite(final_path,images[i])



