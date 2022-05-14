import cv2
import numpy as np
from math import *
import time

image = cv2.imread('0002.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sift(image):

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image,None)
    return

def surf(image):

    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(image,None)

start = time.time()
for i in range(10):
    sift(image)
end = time.time()
print("time elapsed to run sift - ", end - start," seconds")

start = time.time()
for i in range(10):
    surf(image)
end = time.time()
print("time elapsed to run surf - ", end - start," seconds")