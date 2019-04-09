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


def gaussian_plot(sum,mean,sigma,channel):
    x = np.linspace(mean - 3*sigma, mean + 3*sigma, 100)
    plt.plot(x,sum*stats.norm.pdf(x, mean, sigma))
    if channel==1:
        plt.title("Gaussian Mixture - Green Buoy ")
    elif channel==2:
        plt.title("Gaussian Mixture - Red Buoy ")
    elif channel==3:
        plt.title("Gaussian Mixture - Yellow Buoy ")
    plt.xlabel("Value")
    plt.legend()
    plt.ylabel("Frequency")

def gauss_prob_calc(data,mean,sigma):
    norm=1/(np.sqrt(2*np.pi)*sigma)
    probability=norm*np.exp(-(data-mean)**2/(2*sigma**2))
    return probability

def sigma_mean(mat,start,end):
    sum=0
    var_sum=0
    com_mean=0
    for i in range(start,end+1):
        sum=sum+mat[i,0]
        com_mean=com_mean+i*mat[i,0]
    mean=com_mean/sum

    for i in range(start,end+1):
        var_sum=var_sum+(i-mean)**2*mat[i,0]
    sigma=np.sqrt(var_sum/(end-start))

    return sum,mean,sigma

def mean_histo(N,picfold,channel):
    total_hist=0
    for i in range(N+1):
        if picfold==1:
            image=cv2.imread('Green_new/Final%d.jpg' %i)
        if picfold==2:
            image=cv2.imread('Red_new/Final%d.jpg' %i)
        if picfold==3:
            image=cv2.imread('Yellow_new/Final%d.jpg' %i)
        hist = cv2.calcHist([image],[channel],None,[256],[50,256])
        total_hist=total_hist+hist
    Avg_hist=total_hist/(N+1)

    if channel==0:
        plt.plot(Avg_hist,'-b',label="blue")
    elif channel==1:
        plt.plot(Avg_hist,'-g',label="green")
    elif channel==2:
        plt.plot(Avg_hist,'-r',label="red")

    if picfold==1:
        plt.title("Gaussian Histogram - Green Buoy ")
    elif picfold==2:
        plt.title("Gaussian Histogram - Red Buoy ")
    elif picfold==3:
        plt.title("Gaussian Histogram - Yellow Buoy ")

    plt.xlabel("Value")
    plt.legend()
    plt.ylabel("Frequency")
    return Avg_hist

def mean_var(Average,channel):
    gauss_info=np.zeros((2,3))
    if channel==1:
        gauss_info[0,0],gauss_info[0,1],gauss_info[0,2]=sigma_mean(Average,231,255)
        gauss_info[1,0],gauss_info[1,1],gauss_info[1,2]=sigma_mean(Average,120,230)
        plt.plot(hist_gg,'-g',label="green channel")
        gaussian_plot(gauss_info[0,0]*2.5,gauss_info[0,1],gauss_info[0,2],channel_g)
        gaussian_plot(gauss_info[1,0]*1.5,gauss_info[1,1],gauss_info[1,2],channel_g)
    if channel==2:
        plt.plot(hist_rr,'-r',label="red channel")
        gauss_info[0,0],gauss_info[0,1],gauss_info[0,2]=sigma_mean(Average,236,255)
        gaussian_plot(gauss_info[0,0]*2.5,gauss_info[0,1],gauss_info[0,2],channel_r)
        #gauss_info[1,0],gauss_info[1,1],gauss_info[1,2]=sigma_mean(Average,120,230)
    if channel==3:
        plt.plot(hist_yy,'-y',label="yellow channel")
        gauss_info[0,0],gauss_info[0,1],gauss_info[0,2]=sigma_mean(Average,225,255)
        gauss_info[1,0],gauss_info[1,1],gauss_info[1,2]=sigma_mean(Average,120,224)
        gaussian_plot(gauss_info[0,0]*2.5,gauss_info[0,1],gauss_info[0,2],channel_y)
        gaussian_plot(gauss_info[1,0]*1.5,gauss_info[1,1],gauss_info[1,2],channel_y)
    print("Gauss info for channel%d"%channel,gauss_info)
    plt.grid()
    #plt.show()
    return gauss_info

if __name__ == '__main__':

    #No. of training data
    N_r=20
    N_y=37
    N_g=28

    #Channel nos
    channel_b=0
    channel_g=1
    channel_r=2
    channel_y=3

    #Matrix to hold sum, mean and variance info
    y_gauss_info=np.zeros((2,3))
    r_gauss_info=np.zeros((2,3))
    g_gauss_info=np.zeros((2,3))
    #--------------------------------------------------------------------------
    #Plot of histogram and gaussian for each buoy
    #Green
    hist_gb=mean_histo(N_g,channel_g,channel_b)
    hist_gg=mean_histo(N_g,channel_g,channel_g)
    hist_gr=mean_histo(N_g,channel_g,channel_r)
    plt.show()

    g_gauss_info=mean_var(hist_gg,channel_g)
    plt.show()
    #-----------------------------------------------------
    #Red
    hist_rb=mean_histo(N_r,channel_r,channel_b)
    hist_rg=mean_histo(N_r,channel_r,channel_g)
    hist_rr=mean_histo(N_r,channel_r,channel_r)
    plt.show()

    r_gauss_info=mean_var(hist_rr,channel_r)
    plt.show()
    #------------------------------------------------------
    #Yellow
    hist_yb=mean_histo(N_y,channel_y,channel_b)
    hist_yg=mean_histo(N_y,channel_y,channel_g)
    hist_yr=mean_histo(N_y,channel_y,channel_r)
    hist_yy=(hist_yg+hist_yr)/2
    plt.plot(hist_yy,'-y',label="yellow")
    plt.show()
    #------------------------------------------------------
    y_gauss_info=mean_var(hist_yy,channel_y)
    plt.show()
    #-------------------------------------------------------------------------
