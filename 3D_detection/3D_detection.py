import argparse
import numpy as np
import os, sys
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import math
from PIL import Image
import random
import scipy.stats as stats

#from scipy.stats import norm

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import cv2 as cv


def MV_gaussian(point,mean,sigma,D):
    mean_norm=1/(((2*np.pi)**(D/2))*(LA.det(sigma))**(0.5))
    difference=point-mean
    difference=difference.transpose()
    A=np.matmul(difference.transpose(),LA.inv(sigma))
    B=np.abs(np.matmul(A,difference))
    probability=(100000*mean_norm*np.exp(-B/2))
    return probability

def probability_image_green(img,threshold):
    height,width,layers=np.shape(img)
    print(np.shape(img))
    prob_img=0*img
    ext_right=0
    ext_left=1000
    base=0

    mean=np.array([[192.18277387, 248.02716919, 160.15305901],
   [118.34966997, 195.96161778, 119.86500038]])
    weight=np.array([[0.38780729],[0.61219271]])
    sigma=np.array([[[533.97805778, 628.53245143],[ 24.2028958,  618.59601136],[374.47564871, 380.58925531]],
   [[ 24.2028958,  618.59601136],[ 34.46157895, 887.41899556],[-10.27712459,362.22617415]],
   [[374.47564871, 380.58925531],[-10.27712459, 362.22617415],[343.15834175, 262.078583  ]]])

    #print(np.shape(mean))
    #print(np.shape(weight))
    #print(np.shape(sigma))
    for i in range (height):
        for j in range(width):

            #print(np.shape(point),'image point')
            if(i>400 or i<150):
                img[i,j,:]=[0,0,0]
            point=img[i,j,:]
            prob_img[i,j,0]=25500*(weight[0,0]*MV_gaussian(point,mean[0,:],sigma[:,:,0],3)+weight[1,0]*MV_gaussian(point,mean[1,:],sigma[:,:,1],3))
            if(prob_img[i,j,0]>=threshold):
                prob_img[i,j,:]=255
                if (ext_right<j):
                    ext_right=j
                if(abs(ext_right-ext_left)<20):
                    ext_left=ext_right-20
                if (ext_left>j and ext_right-j<20):
                    ext_left=j
                    base=i

            else:
                prob_img[i,j,0]=0
            #print(prob_img[i,j,:],'black and white')
    print(ext_left,'ext_left')
    print(ext_right,'ext_right')
    centre=(ext_left+ext_right)/2
    print(centre,'centre_x')
    print(base-12,'centre_y')
    #centre=np.mat([base-6],[centre])
    flag=0
    if(ext_left==ext_right):
        flag=1
    return prob_img,centre,base-6,flag

def probability_image_red(img,threshold):
    height,width,layers=np.shape(img)
    prob_img=0*img
    ext_right=0
    ext_left=1000
    base=0

    mean=np.array([[118.43316908, 183.83282332, 243.42817752],[140.93158785, 186.2959427,  225.12107122]])
    weight=np.array([[0.70447737],[0.29552263]])
    sigma=np.array([[[ 230.46876748 , 488.78160736],[ 388.37900424,  622.98223896],[ -75.70861234 ,  45.06529594]],[[ 388.37900424,  622.98223896],[ 732.18245657, 1001.46283557],[-148.81014188 , 172.54529128]],
   [[ -75.70861234 ,  45.06529594],[-148.81014188 , 172.54529128],[  49.1459872 ,  332.56742226]]] )

    #print(np.shape(mean))
    #print(np.shape(weight))
    #print(np.shape(sigma))
    for i in range (height):
        for j in range(width):

            #print(np.shape(point),'image point')
            if(i>400 or i<150):
                img[i,j,:]=[0,0,0]
            point=img[i,j,:]
            prob_img[i,j,0]=25500*(weight[0,0]*MV_gaussian(point,mean[0,:],sigma[:,:,0],3)+weight[1,0]*MV_gaussian(point,mean[1,:],sigma[:,:,1],3))
            if(prob_img[i,j,0]>=threshold):
                prob_img[i,j,:]=255
                if (ext_right<j):
                    ext_right=j
                if(abs(ext_right-ext_left)<20):
                    ext_left=ext_right-20
                if (ext_left>j and ext_right-j<20):
                    ext_left=j
                    base=i

            else:
                prob_img[i,j,0]=0
    print(ext_left,'ext_left')
    print(ext_right,'ext_right')
    centre=(ext_left+ext_right)/2
    print(centre,'centre_x')
    print(base-12,'centre_y')
    #centre=np.mat([base-6],[centre])
    flag=0
    if(ext_left==ext_right):
        flag=1
    return prob_img,centre,base-6,flag

def probability_image_yellow(img,threshold):
    height,width,layers=np.shape(img)
    print(np.shape(img))
    prob_img=0*img
    ext_right=0
    ext_left=1000
    base=0

    mean=np.array([[147.5337616,  248.20747064, 237.80292652],[116.22233547, 218.95261565, 219.68868842]])
    weight=np.array([[0.65543956],[0.34456044]])
    sigma=np.array([[[ 932.48845401,  977.39506935],[ -23.68820292,  301.16232229],[ -48.74952313 ,  52.70243927]],
     [[ -23.68820292 , 301.16232229],[   7.89874624,  681.30543371],[   3.76102091 , 811.14484296]],
     [[ -48.74952313,   52.70243927],[   3.76102091,  811.14484296],[  12.15513611, 1231.3601203 ]]])

    print(np.shape(mean))
    print(np.shape(weight))
    print(np.shape(sigma))
    for i in range (height):
        for j in range(width):

            #print(np.shape(point),'image point')
            if(i>400 or i<150):
                img[i,j,:]=[0,0,0]
            point=image[i,j,:]
            prob_img[i,j,0]=25500*(weight[0,0]*MV_gaussian(point,mean[0,:],sigma[:,:,0],3)+weight[1,0]*MV_gaussian(point,mean[1,:],sigma[:,:,1],3))
            if(prob_img[i,j,0]>=threshold):
                prob_img[i,j,:]=255
                if (ext_right<j):
                    ext_right=j
                if(abs(ext_right-ext_left)<20):
                    ext_left=ext_right-20
                if (ext_left>j and ext_right-j<20):
                    ext_left=j
                    base=i

            else:
                prob_img[i,j,0]=0
            #print(prob_img[i,j,:],'black and white')
    print(ext_left,'ext_left')
    print(ext_right,'ext_right')
    centre=(ext_left+ext_right)/2
    print(centre,'centre_x')
    print(base-12,'centre_y')
    #centre=np.mat([base-6],[centre])
    flag=0
    if(ext_left==ext_right):
        flag=1
    return prob_img,centre,base-6,flag

def video(img_array,size):
    video=cv2.VideoWriter('Multivarient.avi',cv2.VideoWriter_fourcc(*'DIVX'), 16.0,size)
    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()


def print_circle(img,dest,old_circle,number):
    img_r = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(img_r,cv.HOUGH_GRADIENT,1,180,param1=100,param2=10,minRadius=13,maxRadius=20)
    index=-1
    min=1000

    if(number==1):
        blue=0
        green=255
        red=0
    if(number==2):
        red=255
        blue=0
        green=0
    if(number==3):
        red=255
        blue=0
        green=255


    if(np.size(circles)!=0):
        if(count==0):
            old_circle=circles[0,1,:]
        for i in range (int(np.size(circles)/3)):
            if(min>circles[0,i,0] and ((circles[0,i,0]-old_circle[0])**2+(circles[0,i,1]-old_circle[1])**2)<=40**2):
                min=circles[0,i,0]
                print((circles[0,index,0]-old_circle[0])**2+(circles[0,index,1]-old_circle[1])**2)
                index=i
                #print('correct')
        print(np.shape(circles))
        if(index!=-1):
            cv2.circle(dest,(circles[0,index,0],circles[0,index,1]),circles[0,index,2],(blue,green,red),2)
            old_circle=circles[0,index,:]
        else:
            cv2.circle(dest,(old_circle[0],old_circle[1]),old_circle[2],(blue,green,red),2)
        return old_circle , dest

def print_circle_green(img,dest,old_circle,number):
    img_r = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(img_r,cv.HOUGH_GRADIENT,1,18,param1=100,param2=10,minRadius=9,maxRadius=15)
    index=-1
    min=1000

    if(number==1):
        blue=0
        green=255
        red=0
    if(number==2):
        red=255
        blue=0
        green=0
    if(number==3):
        red=255
        blue=0
        green=255


    if(np.size(circles)!=0):
        if(count==0):
            old_circle=circles[0,19,:]
        for i in range (int(np.size(circles)/3)):
            if(min>circles[0,i,0] and ((circles[0,i,0]-old_circle[0])**2+(circles[0,i,1]-old_circle[1])**2)<=10**2):
                min=circles[0,i,0]
                print((circles[0,index,0]-old_circle[0])**2+(circles[0,index,1]-old_circle[1])**2)
                index=i
                #print('correct')
        print(np.shape(circles))
        if(index!=-1):
            cv2.circle(dest,(circles[0,index,0],circles[0,index,1]),circles[0,index,2],(blue,green,red),2)
            old_circle=circles[0,index,:]
        else:
            cv2.circle(dest,(old_circle[0],old_circle[1]),old_circle[2],(blue,green,red),2)
        return old_circle , dest

if __name__ == '__main__':
    vidObj = cv2.VideoCapture('detectbuoy.avi')
    count = 0
    success = 1
    img_array=[]
    old_circle_g=[]
    old_circle_r=[]
    old_circle_y=[]
    while (success):
        if (count==0):
            success, image = vidObj.read()
        width,height,layers=image.shape
        size = (height,width)

        image=cv2.GaussianBlur(image,(7,7),0)
        original=image.copy()
        # Green buoy

        threshold_g=180
        img_g,centre_x,centre_y,flag=probability_image_green(original,threshold_g)
        old_circle_g,G_processed=print_circle_green(img_g,image,old_circle_r,1)
        #cv2.imshow('G',G_processed)
        cv2.waitKey(50)
        #red buoy
        threshold_r=1
        img_r,centre_x,centre_y,flag=probability_image_red(original,threshold_r)
        old_circle_r,R_processed=print_circle(img_r,image,old_circle_r,2)
        #cv2.imshow('R',R_processed)
        cv2.waitKey(50)
        # yellow buoy
        threshold_y=3
        img_y,centre_x,centre_y,flag=probability_image_yellow(original,threshold_y)
        old_circle_y,Y_processed=print_circle(img_y,image,old_circle_y,3)
        #cv2.imshow('Y',Y_processed)
        cv2.waitKey(50)

        print(count)
        #cv2.imwrite('%d.jpg' %count,image)
        cv2.imshow('Map',image)
        cv2.waitKey(50)
        count += 1
        #cv2.show()
        img_array.append(image)
        success, image = vidObj.read()


    video(img_array,size)
