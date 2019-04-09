'''
/************************************************************************
 MIT License

 Copyright (c) 2018 Harsh Kakashaniya,Koyal Bhartia, Aalap Rana

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 *************************************************************************/
'''
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


def gaussian_plot(scale,mean,sigma,channel):
    x = np.linspace(mean - 3*sigma, mean + 3*sigma, 100)
    plt.plot(x,scale*stats.norm.pdf(x, mean, sigma))
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

def probability_g_img(img,gauss_info,channel,threshold):
    height,width,layers=np.shape(img)
    prob_img=0*img
    ext_right=0
    ext_left=1000
    base=0
    for i in range (height):
        for j in range(width):
            if(img[i,j,2]>150):
                img[i,j,channel]=0
            if(i>400 or i<50):
                img[i,j,channel]=0

            prob_gauss_1=g_gauss_info[0,0]*gauss_prob_calc(img[i,j,channel],g_gauss_info[0,1],g_gauss_info[0,2])
            prob_gauss_2=g_gauss_info[1,0]*gauss_prob_calc(img[i,j,channel],g_gauss_info[1,1],g_gauss_info[1,2])
            prob_img[i,j,0]=25500*(prob_gauss_1+prob_gauss_2)
            print(prob_img[i,j,0],"green")
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
    centre=(ext_left+ext_right)/2
    flag=0
    if(ext_left==ext_right):
        flag=1
    return prob_img,centre,base-6,flag

def probability_r_img(img,gauss_info,channel,threshold):
    height,width,layers=np.shape(img)
    print(np.shape(img))
    prob_img=0*img
    ext_right=0
    ext_left=1000
    base=0
    for i in range (height):
        for j in range(width):
            if(img[i,j,0]>125):
                img[i,j,channel]=0

            if(img[i,j,1]<100):
                img[i,j,channel]=0

            if(i>400 or i<50):
                img[i,j,channel]=0
            prob_gauss_1=r_gauss_info[0,0]*gauss_prob_calc(img[i,j,channel],r_gauss_info[0,1],r_gauss_info[0,2])
            prob_gauss_2=r_gauss_info[1,0]*gauss_prob_calc(img[i,j,channel],r_gauss_info[0,1],r_gauss_info[0,2])
            prob_img[i,j,0]=25500*(prob_gauss_1+prob_gauss_2)
            print(prob_img[i,j,0],"red")
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
    flag=0
    if(ext_left==ext_right):
        flag=1
    return prob_img,centre,base+6,flag

def probability_y_img(img,gauss_info,channel,threshold):
    height,width,layers=np.shape(img)
    #print(np.shape(img))
    prob_img=0*img
    ext_right=0
    ext_left=1000
    base=0
    for i in range (height):
        for j in range(width):
            if(img[i,j,0]>130):
                img[i,j,1]=0
                img[i,j,2]=0
            #print(np.shape(img_y))
            if(i>400 or i<50):
                img[i,j,1]=0
                img[i,j,2]=0

            prob_gauss_1g=y_gauss_info[0,0]*gauss_prob_calc(img[i,j,1],y_gauss_info[0,1],y_gauss_info[0,2])
            prob_gauss_2g=y_gauss_info[1,0]*gauss_prob_calc(img[i,j,1],y_gauss_info[1,1],y_gauss_info[1,2])
            prob_gauss_3g=y_gauss_info[2,0]*gauss_prob_calc(img[i,j,1],y_gauss_info[2,1],y_gauss_info[2,2])
            prob_gauss_g=prob_gauss_1g+prob_gauss_2g+prob_gauss_3g

            prob_gauss_1r=y_gauss_info[3,0]*gauss_prob_calc(img[i,j,2],y_gauss_info[3,1],y_gauss_info[3,2])
            prob_gauss_2r=y_gauss_info[4,0]*gauss_prob_calc(img[i,j,2],y_gauss_info[4,1],y_gauss_info[4,2])
            prob_gauss_3r=y_gauss_info[5,0]*gauss_prob_calc(img[i,j,2],y_gauss_info[5,1],y_gauss_info[5,2])
            prob_gauss_r=prob_gauss_1r+prob_gauss_2r+prob_gauss_3r

            prob_img[i,j,0]=25500*(prob_gauss_g+prob_gauss_r)
            print(prob_img[i,j,0],"yellow")
            if(prob_img[i,j,0]>=threshold):
                prob_img[i,j,:]=255

                if (ext_left>j):
                    ext_left=j
                if(abs(ext_right-ext_left)>20):
                    ext_right=ext_left+20
                if (ext_right<j and np.abs(ext_left-j)>20):
                    ext_right=j
                    base=i

                '''
                if (ext_right<j):
                    ext_right=j
                if(abs(ext_right-ext_left)<20):
                    ext_left=ext_right-20
                if (ext_left>j and ext_right-j<20):
                    ext_left=j
                    base=i
                '''
            else:
                prob_img[i,j,0]=0
    centre=(ext_left+ext_right)/2
    flag=0
    if(ext_left==ext_right):
        flag=1
    return prob_img,centre,base+6,flag

def video(img_array,size):
    video=cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 16.0,size)
    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()

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
    y_gauss_info=np.zeros((6,3))
    r_gauss_info=np.zeros((2,3))
    g_gauss_info=np.zeros((2,3))
    #--------------------------------------------------------------------------
    # The mean and variance values are obtained from the EM algorithm
    # The 2 rows - 2 gaussians
    # 3 columns - weight, mean and variance
    g_gauss_info[0,:]=[5.91448440e-01,241.15779102,12.56758694]
    g_gauss_info[1,:]=[4.08551560e-01,179.95432209,22.21725008]

    r_gauss_info[0,:]=[8.72802231e-01,241.48904147,8.13059751]
    r_gauss_info[1,:]=[1.27069102e-01,214.27877843,2.20071630e+01]

    y_gauss_info[0,:]=[0.15782688,200.53625028,27.49454724] #green
    y_gauss_info[1,:]=[0.6356152,248.42784556,2.72202924]
    y_gauss_info[2,:]=[0.20655792,235.15379911,9.04619238]
    y_gauss_info[3,:]=[0.15174132,191.24097506,34.97827673]
    y_gauss_info[4,:]=[0.41121641,237.57383173,2.55775231]
    y_gauss_info[5,:]=[0.43704228,239.90370623,7.71799858]
    #--------------------------------------------------------------------------
    #Plot of histogram and gaussian for each buoy
    #Green
    hist_gb=mean_histo(N_g,channel_g,channel_b)
    hist_gg=mean_histo(N_g,channel_g,channel_g)
    hist_gr=mean_histo(N_g,channel_g,channel_r)
    #plt.show()

    plt.plot(hist_gg,'-g',label="green channel")
    gaussian_plot(300,g_gauss_info[0,1],g_gauss_info[0,2],channel_g) # green
    gaussian_plot(200,g_gauss_info[1,1],g_gauss_info[1,2],channel_g) # green
    plt.show()
    #-----------------------------------------------------
    #Red
    hist_rb=mean_histo(N_r,channel_r,channel_b)
    hist_rg=mean_histo(N_r,channel_r,channel_g)
    hist_rr=mean_histo(N_r,channel_r,channel_r)
    plt.show()

    plt.plot(hist_rr,'-r',label="red channel")
    gaussian_plot(300,r_gauss_info[0,1],r_gauss_info[0,2],channel_r) # red
    gaussian_plot(200,r_gauss_info[1,1],r_gauss_info[1,2],channel_r) # red
    plt.show()
    #------------------------------------------------------
    #Yellow
    hist_yb=mean_histo(N_y,channel_y,channel_b)
    hist_yg=mean_histo(N_y,channel_y,channel_g)
    hist_yr=mean_histo(N_y,channel_y,channel_r)
    hist_yy=(hist_yg+hist_yr)/2
    plt.plot(hist_yy,'-y',label="yellow")
    plt.show()

    plt.plot(hist_yy,'-y',label="yellow channel")
    gaussian_plot(300,y_gauss_info[0,1],y_gauss_info[0,2],channel_y) # yellow
    gaussian_plot(300,y_gauss_info[1,1],y_gauss_info[1,2],channel_y) # yellow
    gaussian_plot(300,y_gauss_info[2,1],y_gauss_info[2,2],channel_y)
    gaussian_plot(300,y_gauss_info[3,1],y_gauss_info[3,2],channel_y)
    gaussian_plot(300,y_gauss_info[4,1],y_gauss_info[4,2],channel_y)
    gaussian_plot(300,y_gauss_info[5,1],y_gauss_info[5,2],channel_y)
    plt.show()
    #-------------------------------------------------------------------------
    # The process of calculation of probabilily and detection
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
        #---------------------------------------------------------------
        # Green buoy
        threshold_g=188
        img_g,centre_x,centre_y,flag=probability_g_img(original,g_gauss_info,channel_g,threshold_g)
        if(centre_x!=500):
            image=cv2.circle(image,(int(centre_x),int(centre_y)),12,(0,255,0),2)
        #cv2.imshow('Green',img_g)
        #---------------------------------------------------------------
        # red buoy
        threshold_r=252
        img_r,centre_x,centre_y,flag=probability_r_img(image.copy(),r_gauss_info,channel_r,threshold_r)
        if(centre_x!=500):
            image=cv2.circle(image,(int(centre_x),int(centre_y)),12,(0,0,255),2)
        #cv2.imshow('Red',img_r)
        #---------------------------------------------------------------
        # yellow buoy
        threshold_y=241
        img_y,centre_x,centre_y,flag=probability_y_img(original,y_gauss_info,channel_y,threshold_y)
        if(centre_x!=500):
            image=cv2.circle(image,(int(centre_x),int(centre_y)),12,(0,255,255),2)
        #cv2.imshow('Yellow',img_y)

        cv2.imshow('Map',image)
        cv2.waitKey(50)
        img_array.append(image)
        success, image = vidObj.read()

    video(img_array,size)
