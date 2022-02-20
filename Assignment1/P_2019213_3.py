import numpy as np
import cv2
from math import *

video = cv2.VideoCapture("shahar_walk.avi")

allFrames = []
while(video.isOpened()):
    ret, frame = video.read()
    if(not ret):
        break
    allFrames.append(np.array(frame))
    

video.release()
cv2.destroyAllWindows()

n,m,a = allFrames[0].shape

medianFrame = np.median(allFrames,axis=0).astype(dtype='uint8')
cv2.imshow('Median frame',medianFrame)
cv2.waitKey()

circleImage=[]

for k in range(len(allFrames)):
    image=np.zeros((n,m),dtype='uint8')
    difference = cv2.absdiff(allFrames[k],medianFrame)
    for i in range(n):
        for j in range(m):
            image[i][j] = (difference[i][j][0]+difference[i][j][1]+difference[i][j][2])/3

    mapFrame=np.where(image>35)
    finalFrame = np.copy(allFrames[k])
    for i in range(len(mapFrame[0])):
        finalFrame[mapFrame[0][i]][mapFrame[1][i]][0]=0
        finalFrame[mapFrame[0][i]][mapFrame[1][i]][1]=255
        finalFrame[mapFrame[0][i]][mapFrame[1][i]][2]=0

    minX,maxX,minY,maxY = (min(mapFrame[0]),max(mapFrame[0]),min(mapFrame[1]),max(mapFrame[1]))

    radius = max((maxX-minX)//2,(maxY-maxY)//2)
    center = ((minY+maxY)//2,(minX+maxX)//2)

    image = cv2.circle(allFrames[k],center,radius,(0,0,255),1)
    circleImage.append(image)

output = cv2.VideoWriter('circled_shahar_walk.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (m,n))

for i in circleImage:
    output.write(i)
output.release()
