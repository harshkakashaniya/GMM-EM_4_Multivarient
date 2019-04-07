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


def gaussian_plot(sum,mean,sigma):
    x = np.linspace(mean - 3*sigma, mean + 3*sigma, 100)
    plt.plot(x,sum*1.5*stats.norm.pdf(x, mean, sigma))

def gauss_prob_calc(data,mean,sigma):
    norm=1/(np.sqrt(2*np.pi)*sigma)
    probability=norm*np.exp(-(data-mean)**2/(2*sigma**2))
    return probability

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

    mean=np.array([[192.17901368,248.02617201,160.15104284],[118.3466913,  195.95846905, 119.86335246]])
    weight=np.array([[0.38785174],[0.61214826]])
    sigma=np.array([[[534.05901644, 628.44398161],[ 24.23222538, 618.5121379 ],[374.51345936, 380.54004468]],[[ 24.23222538, 618.5121379 ],[ 34.47246061, 887.34297952]
    ,[-10.26129176, 362.18283792]],[[374.51345936, 380.54004468],[-10.26129176, 362.18283792],[343.17031505, 262.05017765]]] )

    print(np.shape(mean))
    print(np.shape(weight))
    print(np.shape(sigma))
    for i in range (height):
        for j in range(width):
            point=image[i,j,:]
            #print(np.shape(point),'image point')
            if(i>400 or i<50):
                img[i,j,:]=[0,0,0]
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
    print(np.shape(img))
    prob_img=0*img
    ext_right=0
    ext_left=1000
    base=0

    mean=np.array([[118.43556246, 183.83410373, 243.4273098 ],[140.93893813, 186.29431775, 225.11251407]])
    weight=np.array([[0.70464883],[0.29535117]])
    sigma=np.array([[[ 230.53360442,  488.73990265],[ 388.45165311,  622.99572961],[ -75.71243374,   45.23257699]],
     [[ 388.45165311,  622.99572961],[ 732.27766769, 1001.40351946],[-148.80271571,  172.67926355]],
     [[ -75.71243374,   45.23257699],[-148.80271571,  172.67926355],[  49.14959307,  332.60451414]]] )

    print(np.shape(mean))
    print(np.shape(weight))
    print(np.shape(sigma))
    for i in range (height):
        for j in range(width):
            point=image[i,j,:]
            #print(np.shape(point),'image point')
            if(i>400 or i<50):
                img[i,j,:]=[0,0,0]
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

def probability_image_yellow(img,threshold):
    height,width,layers=np.shape(img)
    print(np.shape(img))
    prob_img=0*img
    ext_right=0
    ext_left=1000
    base=0

    mean=np.array([[147.53462342, 248.20743605, 237.80296907],[116.21785446, 218.95002672, 219.68696368]])
    weight=np.array([[0.65547083],[0.34452917]])
    sigma=np.array([[[ 932.50818119,  977.16994556],[ -23.68497745,  301.05662236],[ -48.74672225,   52.62260426]],[[ -23.68497745,  301.05662236],[   7.90005073,  681.29024021],[   3.7622569,   811.16648857]],
       [[ -48.74672225,   52.62260426],[   3.7622569,   811.16648857],[  12.15761285, 1231.43333389]]])

    print(np.shape(mean))
    print(np.shape(weight))
    print(np.shape(sigma))
    for i in range (height):
        for j in range(width):
            point=image[i,j,:]
            #print(np.shape(point),'image point')
            if(i>400 or i<50):
                img[i,j,:]=[0,0,0]
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


if __name__ == '__main__':
    vidObj = cv2.VideoCapture('detectbuoy.avi')
    count = 0
    success = 1
    img_array=[]
    while (success):
        if (count==0):
            success, image = vidObj.read()
        width,height,layers=image.shape
        size = (height,width)
        count += 1
        image=cv2.GaussianBlur(image,(7,7),0)
        original=image.copy()
        # Green buoy

        threshold_g=150
        img_g,centre_x,centre_y,flag=probability_image_green(original,threshold_g)
        if(centre_x!=500):
            image=cv2.circle(image,(int(centre_x),int(centre_y)),12,(0,255,0),2)
        cv2.imshow('Map1',img_g)

        # red buoy
        threshold_r=258
        img_r,centre_x,centre_y,flag=probability_image_red(original,threshold_r)
        if(centre_x!=500):
            image=cv2.circle(image,(int(centre_x),int(centre_y)),12,(0,0,255),2)
        cv2.imshow('Map2',img_r)
        # yellow buoy

        #threshold_y=250
        #img_y,centre_x,centre_y,flag=probability_image_yellow(original,threshold_y)
        #if(centre_x!=500):
        #    image=cv2.circle(image,(int(centre_x),int(centre_y)),12,(0,255,255),2)
        #cv2.imshow('Map3',img_y)




        print(count)
        #cv2.imwrite('%d.jpg' %count,image)
        cv2.imshow('Map',image)
        cv2.waitKey(50)
        #cv2.show()
        #img_array.append(Final)
        success, image = vidObj.read()
