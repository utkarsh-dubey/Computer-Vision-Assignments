import numpy as np
import cv2
from math import *
import csv

image = cv2.imread('horse.jpg')
cv2.imshow('Input image',image)
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('Greyscale input image',image)

inputImage = np.array(image)
n,m = inputImage.shape

minTTS = inf
threshold = 0

csvFile = open('values.csv','w',newline='')
csv.writer(csvFile).writerow(['Threshold','TTS'])

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
    csv.writer(csvFile).writerow([thres,TTS])

csvFile.close()

binaryMask = np.zeros(inputImage.shape,dtype='uint8')
binaryMask[inputImage<threshold]=0
binaryMask[inputImage>=threshold]=255

cv2.imshow("Binary Masked Image",binaryMask)
cv2.imwrite("Question2_output.png",binaryMask)
print("Otsu threshold - ",threshold)
cv2.waitKey()
cv2.destroyAllWindows()

