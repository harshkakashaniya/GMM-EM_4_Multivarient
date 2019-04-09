
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

def sigma_mean(mat,start,end):
    sum=0
    var_sum=0
    Harsh=0
    for i in range(start,end+1):
        sum=sum+mat[i,0]
        Harsh=Harsh+i*mat[i,0]
    mean=Harsh/sum

    for i in range(start,end+1):
        var_sum=var_sum+(i-mean)**2*mat[i,0]

    return sum,mean,np.sqrt(var_sum/(end-start))



def mean_histo(N,channel):
    total_hist=0
    for i in range(N+1):
        image=cv2.imread('Final%d.jpg' %i)
        print(np.shape(image),'In histo')
        hist = cv2.calcHist([image],[channel],None,[256],[50,256])
        total_hist=total_hist+hist
    Avg_hist=total_hist/(N+1)
    if (channel==0):
        plt.plot(Avg_hist,'-b',label="blue")
    if (channel==1):
        plt.plot(Avg_hist,'-g',label="green")
    if (channel==2):
        plt.plot(Avg_hist,'-r',label="red")
    plt.title("Gaussian Histogram")
    plt.xlabel("Value")
    plt.legend()
    plt.ylabel("Frequency")
    #plt.show()
    return Avg_hist

def mean_var(Average):
    ##------------------------------------------
    sum_right,mean_right,sigma_right=sigma_mean(Average,231,255)
    sum_mid,mean_mid,sigma_mid=sigma_mean(Average,181,230)
    sum_left,mean_left,sigma_left=sigma_mean(Average,130,180)
    print(sum_right,mean_right,sigma_right)
    print(sum_left,mean_left,sigma_left)

    x_right = np.linspace(mean_right - 3*sigma_right, mean_right + 3*sigma_right, 100)
    plt.plot(x_right,sum_right*2.5*stats.norm.pdf(x_right, mean_right, sigma_right))
    x_left = np.linspace(mean_left - 3*sigma_left, mean_left + 3*sigma_left, 100)
    plt.plot(x_left,sum_left*1.5*stats.norm.pdf(x_left, mean_left, sigma_left))
    x_mid = np.linspace(mean_mid - 3*sigma_mid, mean_mid + 3*sigma_mid, 100)
    plt.plot(x_mid,sum_mid*1.5*stats.norm.pdf(x_mid, mean_mid, sigma_mid))
    plt.grid()
    plt.show()

    return sum_right,mean_right,sigma_right,sum_left,mean_left,sigma_left,sum_mid,mean_mid,sigma_mid

def probability_img(img,mean_left,Sigma_left,mean_right,Sigma_right,sum_mid,mean_mid,sigma_mid,channel,threshold):
    height,width,layers=np.shape(img)
    print(np.shape(img))
    prob_img=0*img[:,:,0]
    norm_left=1/(np.sqrt(2*np.pi)*sigma_left)
    norm_mid=1/(np.sqrt(2*np.pi)*sigma_mid)
    norm_right=1/(np.sqrt(2*np.pi)*sigma_right)
    norm=(norm_left+norm_right+norm_mid)/3
    for i in range (height):
        for j in range(width):
            prob_img[i,j]=255*(norm_left*np.exp(-(img[i,j,channel]-mean_left)**2/(2*Sigma_left**2))+norm_right*np.exp(-(img[i,j,channel]-mean_right)**2/(2*Sigma_right**2))+norm_mid*np.exp(-(img[i,j,channel]-mean_mid)**2/(2*sigma_mid**2)))
            print(prob_img[i,j])
            if(prob_img[i,j]==255):
                prob_img[i,j]=0
            else:
                prob_img[i,j]=255
            #if(prob_img[i,j]<0.3*255 and prob_img[i,j]>0.75*255 ):
            #    prob_img[i,j]=255
            #print(img[i,j,channel],'intensity')
            #print(prob_img[i,j],i,j)
    return prob_img


##

Averageg=mean_histo(26,1)

sum_right,mean_right,sigma_right,sum_left,mean_left,sigma_left,sum_mid,mean_mid,sigma_mid=mean_var(Averageg)

#image=cv2.imread('../../11.jpg')
#img=probability_img(image,mean_left,sigma_left,mean_right,sigma_right,sum_mid,mean_mid,sigma_mid,1,7500)
#cv2.imshow('Map',img)
#cv2.waitKey()
#cv2.show()


#mean_yellow=(meanr+meang)/2
#sigma_yellow=(sigmar+sigmag)/2


#x = np.linspace(mean_yellow - 3*sigma_yellow, mean_yellow + 3*sigma_yellow, 100)
#plt.plot(x, stats.norm.pdf(x, mean_yellow, sigma_yellow))
#plt.axis([0,200,-2,2])
#plt.grid()
#plt.show()


#Averageb=mean_histo(5,0)

#meanb,varb=mean_var(Averageb)

'''
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

    lower = np.array([meanb-1*sigmab, meang-1*sigmag, meanr-1*sigmar])
    upper = np.array([meanb+1*sigmab, meang+1*sigmag, meanr+1*sigmar])
    mask = cv2.inRange(image, lower, upper)
    print('Frame processing index')

    print(count)
    #cv2.imwrite('%d.jpg' %count,image)
    cv2.imshow('Map',mask)
    cv2.waitKey(50)
    #cv2.show()
    #img_array.append(Final)
    success, image = vidObj.read()
'''
