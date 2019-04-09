
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


def gaussian_plot(mean,sigma):
    x = np.linspace(mean - 3*sigma, mean + 3*sigma, 100)
    plt.plot(x,200*stats.norm.pdf(x, mean, sigma))

def gauss_prob_calc(data,mean,sigma):
    norm=1/(np.sqrt(2*np.pi)*sigma)
    probability=mean*np.exp(-(data-mean)**2/(2*sigma**2))
    return probability


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
    sigma=np.sqrt(var_sum/(end-start))

    return sum,mean,sigma

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
    gauss_info[0,0],gauss_info[0,1],gauss_info[0,2]=sigma_mean(Average,231,255)
    gauss_info[1,0],gauss_info[1,1],gauss_info[1,2]=sigma_mean(Average,181,230)
    gauss_info[2,0],gauss_info[2,1],gauss_info[2,2]=sigma_mean(Average,130,180)
    print(gauss_info)
    gaussian_plot(241,12.56)
    gaussian_plot(179.95,22.21)

    #gaussian_plot(gauss_info[0,0],gauss_info[0,1],gauss_info[0,2])
    #gaussian_plot(gauss_info[1,0],gauss_info[1,1],gauss_info[1,2])
    #gaussian_plot(gauss_info[2,0],gauss_info[2,1],gauss_info[2,2])
    plt.grid()
    plt.show()

    return gauss_info

def probability_img(image,gauss_info,channel,threshold):
    height,width,layers=np.shape(img)
    print(np.shape(img))
    prob_img=0*img[:,:,0]
    norm_left=1/(np.sqrt(2*np.pi)*sigma_left)
    norm_mid=1/(np.sqrt(2*np.pi)*sigma_mid)
    norm_right=1/(np.sqrt(2*np.pi)*sigma_right)
    norm=(norm_left+norm_right+norm_mid)/3
    for i in range (height):
        for j in range(width):
            prob_img[i,j]=255*(gauss_prob_calc(dataimg[i,j,channel],gauss_info[0,1],gauss_info[0,2])+gauss_prob_calc(data,gauss_info[1,1],gauss_info[1,2])+gauss_prob_calc(data,gauss_info[2,1],gauss_info[2,2]))
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


if __name__ == '__main__':

    channel=1
    threshold=7500
    gauss_info=np.zeros((3,3))
    histg=mean_histo(26,1)
    gauss_info=mean_var(histg)
    '''
    image=cv2.imread('../../11.jpg')
    img=probability_img(image,gauss_info,channel,threshold)
    cv2.imshow('Map',img)
    cv2.waitKey()
    cv2.show()
    '''
