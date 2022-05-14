import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import *
from sklearn.cluster import DBSCAN

image = cv2.imread('./0002.jpg')
m,n,_=image.shape
newImage = np.reshape(image,[m*n,3])
for e in [4,8,12,14]:
    for samples in [10,20,30]:
        db = DBSCAN(eps=e, min_samples=samples, metric = 'euclidean', algorithm ='auto')
        db.fit(newImage)
        labels = db.labels_

        plt.figure("Plot for epi = "+str(e)+" and min_samples = "+str(samples))
        plt.imshow(np.reshape(labels,[m,n]))
        plt.savefig("Plot for epi = "+str(e)+" and min_samples = "+str(samples)+".jpg")
        
plt.show()
