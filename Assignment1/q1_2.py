import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import pandas as pa

img=cv2.imread("Segmented_BigTree.png",1)
M,N,a=np.shape(img)

img_orig=cv2.imread("Big_Tree_kmeans.png",1)
region_colors=np.unique(img.reshape((img.shape[1]*img.shape[0],3)), axis=0)

print("No. of segments: ",len(region_colors))

saliency=np.zeros(len(region_colors))
Dist_segments=np.zeros((len(region_colors),len(region_colors)))
segments=[]

#find all the pixels belonging to a region(segment)
for i in range(len(region_colors)):
    these=np.where((img[:,:,0] == region_colors[i][0]) & (img[:,:,1] == region_colors[i][1]) & (img[:,:,2] ==region_colors[i][2]))
    segments.append(these)


   
for i in range(len(segments)):
    col_in_i=[]
    # finding all the colors and unique ones in a segment
    for t in range(len(segments[i][0])):
        this=img_orig[segments[i][0][t]][segments[i][1][t]]
        to_put=[this[0],this[1],this[2]]
        # col_in_i.append(np.array(img_orig[segments[i][0][t]][segments[i][1][t]]))
        col_in_i.append(to_put)
    unique_in_i=np.unique(col_in_i,axis=0)
    col_in_i=np.array(col_in_i)
    
    # finding all the colors and unique ones in the other segment 
    for j in range(len(segments)):        
        col_in_j=[]
        for t in range(len(segments[j][0])):
            this=img_orig[segments[j][0][t]][segments[j][1][t]]
            to_put=[this[0],this[1],this[2]]
            # col_in_j.append(img_orig[segments[j][0][t]][segments[j][1][t]])
            col_in_j.append(to_put)
        unique_in_j=np.unique(col_in_j,axis=0)
        col_in_j=np.array(col_in_j)
        
        # print("U:",unique_in_i)
        # print("A:",col_in_i)
        
        Dc=0
        #summation over all the unique colors in both the segments
        for k in range(len(unique_in_i)):
            for l in range(len(unique_in_j)):
                # print(np.where((col_in_j == unique_in_j[l])))
                fi=len(np.where((col_in_i == unique_in_i[k]))[0])/len(col_in_i)
                fj=len(np.where((col_in_j == unique_in_j[l]))[0])/len(col_in_j)
                
                # fi=len(np.where((col_in_i[0] == unique_in_i[k][0]) & (col_in_i[1] == unique_in_i[k][1]) & (col_in_i[2] == unique_in_i[k][2])))/len(col_in_i)
                # fj=len(np.where((col_in_j[0] == unique_in_j[l][0]) & (col_in_j[1] == unique_in_j[l][1]) & (col_in_j[2] == unique_in_j[l][2])))/len(col_in_j)
                Dc+=(fi*fj*(((unique_in_i[k][0]-unique_in_j[l][0])**2+(unique_in_i[k][1]-unique_in_j[l][1])**2+(unique_in_i[k][2]-unique_in_j[l][2])**2)**(1/2)))
                # print(fi,fj)
        
        Dist_segments[i][j]=Dc #distance between segments i and j

for i in range(len(saliency)):
    sums=0
    for j in range(len(saliency)):
        sums+=((len(segments[j][0])/M*N)*Dist_segments[i][j]) #weight*distance between segments
    saliency[i]=sums

output_img=np.zeros((M,N),np.uint8)
max_sal=max(saliency)
for i in range(len(segments)):
       for j in range(len(segments[i][0])):
           output_img[segments[i][0][j]][segments[i][1][j]]=(saliency[i]/max_sal)*255
           
cv2.imshow("output",output_img)
cv2.waitKey()
cv2.imwrite("eqn5.png",output_img)
