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

def gaussian(point,mean,sigma):
    mean_norm=1/(np.sqrt(2*np.pi)*sigma)
    probability=mean_norm*np.exp(-(point-mean)**2/(2*sigma**2))
    return probability

def Initialize_parameter(number):
    mean=np.zeros((number,1))
    sigma=np.zeros((number,1))
    weight=np.zeros((number,1))

def E_step(data,weight,mean,sigma):
    classified=np.zeros((len(data),len(mean)))
    for i in range (len(data)):
        for j in range (len(mean)):
            classified[i,j]=gaussian(data[i],mean[j],sigma[j])*weight[j]

    for j in range (len(data)):
        sum=np.sum(classified[j,:])
        for i in range (len(mean)):
            classified[j,i]=classified[j,i]/sum
    return classified

def sigma_mean(mat,start,end,weight):
    sum=0
    var_sum=0
    Harsh=0
    r=0.0001

    for i in range(start,end+1):
        mat[i]=mat[i]*weight
        sum=sum+mat[i]
        Harsh=Harsh+i*mat[i]
    mean=Harsh +r

    for i in range(start,end+1):
        var_sum=var_sum+(i-mean)**2*mat[i]

    return sum,mean,np.sqrt(var_sum/(end-start))+r

def M_step(classified_data,data):
    r=0.01

    new_m=np.zeros((len(classified_data[0]),1))
    new_sigma=np.zeros((len(classified_data[0]),1))
    new_weight=np.zeros((len(classified_data[0]),1))
    sum_weight=0

    for j in range(len(classified_data[0])):
        sum_weight=sum_weight+np.sum(classified_data[:,j])
    print(sum_weight)
    for j in range(len(classified_data[0])):
        new_weight[j,0]=np.sum(classified_data[:,j])/sum_weight
    for i in range(len(classified_data[0])):
        m=0
        for j in range(len(classified_data)):
            m = m+data[j]*classified_data[j,i]
        new_m[i,0]=m/np.sum(classified_data[:,i])
    #print(new_m,'new m')


    for j in range(len(classified_data[0])):
        sigma=0
        for i in range(len(classified_data)):
            difference=(data[i]-new_m[j,0])
            sigma=sigma+classified_data[i,j]*difference**2
        new_sigma[j,0]=np.sqrt(sigma/np.sum(classified_data[:,j]))
    #print(new_sigma,'new_sigma')

    return new_weight,new_m,new_sigma

def mean_histo(N,channel):
    total_hist=0
    for i in range(N+1):
        image=cv2.imread('Final%d.jpg' %i)
        #print(np.shape(image),'In histo')
        hist = cv2.calcHist([image],[channel],None,[256],[0,256])
        total_hist=total_hist+hist
    Avg_hist=total_hist/(N+1)
    return Avg_hist


def data_generator(N,channel):
    data=[]
    total_hist=0
    for i in range(N+1):
        image=cv2.imread('Final%d.jpg' %i)
        for i in range (len(image)):
            for j in range (len(image[0])):
                if int(image[i,j,channel])>50:
                    print(image[i,j,channel])
                    data=np.append([data],[int(image[i,j,channel])])
    data=np.array(data)
    data=data.transpose()
    print(min(data),'minimum value')
    return data
#main

data=data_generator(37,1)
#data2=data_generator(37,2)
#print(len(data))
#print(data)
Initialize_parameter(3)
mean=[200,240,255]
sigma=[10,15,20]
weight=[1/3,1/3,1/3]

while(True):
#for i in range(100):
    old_mean=mean
    old_sigma=sigma
    #print(mean,sigma)
    classified_data=E_step(data,weight,mean,sigma)
    print(np.sum(classified_data[0,:]),'sum')
    print(np.sum(classified_data[50,:]),'sum')
    weight,mean,sigma=M_step(classified_data,data)
    print(weight,'weight')
    print(mean,'mean')
    print(sigma,'sigma')
    #break
    #if math.isnan(mean).any() or math.isnan(sigma).any():
    #    break
    if(np.abs(old_mean[0]-mean[0])<=0.01 and np.abs(old_mean[1]-mean[1])<=0.01):
        if(np.abs(old_sigma[0]-sigma[0])<=0.01 and np.abs(old_sigma[1]-sigma[1])<=0.01):
            break

plt.plot(data,'-g',label="blue")
x_right = np.linspace(mean[0] - 3*sigma[0], mean[0] + 3*sigma[0], 100)
plt.plot(x_right,100*stats.norm.pdf(x_right, mean[0], sigma[0]))

x_right = np.linspace(mean[1] - 3*sigma[1], mean[1] + 3*sigma[1], 100)
plt.plot(x_right,150*stats.norm.pdf(x_right, mean[1], sigma[1]))
plt.grid()
#plt.axis([-2,20,0,15])
plt.show()

# print(classified_data)
