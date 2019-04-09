import argparse
import numpy as np
import os, sys
import numpy as np
from matplotlib import style
from numpy import linalg as LA
from matplotlib import pyplot as plt
import math
from PIL import Image
import random
import scipy.stats as stats

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import cv2 as cv

def print_circle(img,old_circle):
    img_r = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(img_r,cv.HOUGH_GRADIENT,1,18,param1=100,param2=10,minRadius=9,maxRadius=15)
    index=-1
    min=1000
    if(count==0):
        old_circle=circles[0,38,:]
    for i in range (int(np.size(circles)/3)):
        if(min>circles[0,i,0] and ((circles[0,i,0]-old_circle[0])**2+(circles[0,i,1]-old_circle[1])**2)<=10**2):
            min=circles[0,i,0]
            print((circles[0,index,0]-old_circle[0])**2+(circles[0,index,1]-old_circle[1])**2)
            index=i
            #print('correct')
    print(np.shape(circles))
    if(index!=-1):
        cv2.circle(img,(circles[0,index,0],circles[0,index,1]),circles[0,index,2],(0,255,0),2)
        old_circle=circles[0,index,:]
    else:
        cv2.circle(img,(old_circle[0],old_circle[1]),old_circle[2],(0,255,0),2)

    cv2.imshow('A',img)
    return old_circle

def video(img_array,size):
    video=cv2.VideoWriter('Multivarient.avi',cv2.VideoWriter_fourcc(*'DIVX'), 16.0,size)
    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()

N=17
count=0
old_circle=[]
img_array=[]
for i in range(4,N):
    img=cv2.imread('%d.jpg' %i)
    size=(640,480)
    #circles = np.uint16(np.around(circles))
    #circles=circles[0,0,:]
    old_circle=print_circle(img,old_circle)
    count=count+1

    cv2.waitKey(100)
    img_array.append(img)

video(img_array,size)
