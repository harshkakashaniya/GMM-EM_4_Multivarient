
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
        #print(np.shape(image),'In histo')
        hist = cv2.calcHist([image],[channel],None,[256],[50,256])
        total_hist=total_hist+hist
    Avg_hist=total_hist/(N+1)
    if (channel==0):
        plt.plot(Avg_hist,'-b',label="blue")
    if (channel==1):
        plt.plot(Avg_hist,'-g',label="green")
    if (channel==2):
        plt.plot(Avg_hist,'-r',label="red")
    plt.title("Gaussian Histogram - Red Buoy")
    plt.xlabel("Value")
    plt.legend()
    plt.ylabel("Frequency")
    plt.show()
    return Avg_hist

def mean_var(Average):
    ##------------------------------------------
    sum_right,mean_right,sigma_right=sigma_mean(Average,236,255)
    print(sum_right,mean_right,sigma_right,"mean guy")

    x_right = np.linspace(mean_right - 3*sigma_right, mean_right + 3*sigma_right, 100)
    plt.plot(x_right,sum_right*2.5*stats.norm.pdf(x_right, mean_right, sigma_right))
    #plt.grid()
    #plt.show()
    return sum_right,mean_right,sigma_right

def probability_img(img,mean_right,Sigma_right,channel,threshold):
    height,width,layers=np.shape(img)
    print(np.shape(img))
    prob_img=0*img[:,:,0]
    norm=1/(np.sqrt(2*math.pi)*Sigma_right)
    for i in range(height):
        for j in range(width):
            if(img[i,j,0]>180):
                img[i,j,channel]=0

    #print(norm)
    for i in range (height):
        for j in range(width):
            prob_img[i,j]=25500*norm*np.exp(-(img[i,j,channel]-mean_right)**2/(2*Sigma_right**2))
            print(prob_img[i,j],'Image prob')
            if(prob_img[i,j]>(threshold)):
                prob_img[i,j]=255
            else:
                prob_img[i,j]=0
            print(prob_img[i,j])
    return prob_img


##



Averager=mean_histo(20,2)
#Averageg=mean_histo(20,1)
#Averageb=mean_histo(20,0)


sum_right,mean_right,sigma_right=mean_var(Averager)
#plt.show()

image=cv2.imread('images/0.jpg')
img=probability_img(image,mean_right,sigma_right,2,246)
cv2.imshow('Map',img)
cv2.waitKey()
cv2.show()


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
