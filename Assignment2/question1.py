import cv2
import glob
import numpy as np
from math import *
from copy import deepcopy
import scipy.stats
from scipy.integrate import quad
import csv

def gaussian(x, mu, sigma):
    return (1/(sigma*sqrt(2*pi)))*np.exp(-(x/sigma - mu/sigma)**2)

def otsu(inputImage):
    minTTS = inf
    threshold = 0
    n,m = inputImage.shape
    for thres in range(1,256):
        less=[]
        more=[]
        for i in range(n):
            for j in range(m):
                if(inputImage[i][j]<thres):
                    less.append(inputImage[i][j])
                    continue
                more.append(inputImage[i][j])
        TTS = np.var(less)*len(less) + np.var(more)*len(more)
        if(TTS<minTTS):
            minTTS=TTS
            threshold=thres
    return threshold



def separate(saliencyMap):
    
    saliencyMap = (saliencyMap/np.max(saliencyMap))*255
    otsuThreshold = otsu(saliencyMap.astype(int))
    print("thres",otsuThreshold)
    mask = deepcopy(saliencyMap)
    mask[mask<otsuThreshold] = 0
    mask[mask>=otsuThreshold] = 1
    foregroundMask = mask
    backgroundMask = 1 - foregroundMask

    foregroundMap = saliencyMap*foregroundMask
    foregroundMap/=np.max(foregroundMap)
    backgroundMap = saliencyMap*backgroundMask
    backgroundMap/=np.max(backgroundMap)

    foregroundMu = np.mean(foregroundMap[foregroundMap>0])
    backgroundMu = np.mean(backgroundMap[backgroundMap>0])

    foregroundSigma = np.std(foregroundMap[foregroundMap>0])
    backgroundSigma = np.std(backgroundMap[backgroundMap>0])
    print(foregroundSigma,backgroundSigma)

    z = (backgroundMu*foregroundSigma**2 - foregroundMu*backgroundSigma**2)/(foregroundSigma**2-backgroundSigma**2) + (foregroundSigma*backgroundSigma/(foregroundSigma**2-backgroundSigma**2))*((foregroundMu-backgroundMu)**2 - 2*(foregroundSigma**2-backgroundSigma**2)*(log(backgroundSigma)-log(foregroundSigma)))**(1/2)
    g=255
    ls = quad(gaussian, 0, z, args=(foregroundMu,foregroundSigma))[0] + quad(gaussian, z, 1, args=(backgroundMu,backgroundSigma))[0]
    phi = 1/(1 + log(1 + g*ls,10))
    
    return phi

allNames = glob.glob('dl/*.png')
imagesDl = [cv2.imread(file,0) for file in glob.glob('dl/*.png')]
imagesNonDl = [cv2.imread(file,0) for file in glob.glob('nondl/*.png')]

csvFile = open('question1.csv','w',newline='')
csv.writer(csvFile).writerow(['imageName','dlValues','nonDlValues'])
print(allNames)
for i in range(len(allNames)):
    # cv2.imshow("image",imagesDl[i])
    # cv2.waitKey()
    dlValue = separate(imagesDl[i])
    nonDlValue = separate(imagesNonDl[i])
    print(dlValue,nonDlValue)
    csv.writer(csvFile).writerow([allNames[i],dlValue,nonDlValue])

csvFile.close()
