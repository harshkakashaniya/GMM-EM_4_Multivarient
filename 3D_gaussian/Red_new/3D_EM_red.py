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
from scipy.stats import multivariate_normal as mult
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2


def MV_gaussian(point,mean,sigma,D):
    mean_norm=1/(((2*np.pi)**(D/2))*(LA.det(sigma))**(0.5))
    #print(mean_norm,'mean_norm')

    difference=point-mean
    #print(difference,'difference')
    difference=difference.transpose()
    A=np.matmul(difference.transpose(),LA.inv(sigma))
    B=np.abs(np.matmul(A,difference))
    #print(LA.inv(sigma),'inverse')
    #print(B,'B')
    #print(A,'A')
    probability=mean_norm*np.exp(-B/2)
    return probability

def Initialize_parameter(K,D):
    mean=np.array(np.zeros((K,D)))
    sigma=np.array(np.zeros((D,D,K)))
    weight=np.array(np.zeros((K,1)))
    return mean,sigma,weight

def E_step(data,weight,mean,sigma,D):
    classified=np.zeros((len(data),len(mean)))
    check=0
    for i in range (len(data)):
        for j in range (len(mean)):
            classified[i,j]=100000*MV_gaussian(data[i,:],mean[j,:],sigma[:,:,j],D)*weight[j,:]
    for j in range (len(data)):
        check=check+math.log(np.sum(classified[j,:]))


    for j in range (len(data)):
        sum=np.sum(classified[j,:])
        for i in range (len(mean)):
            classified[j,i]=classified[j,i]/sum
    print(classified)

    return classified,check

def M_step(classified_data,data,D):
    new_m=np.zeros((len(classified_data[0]),D))
    print(len(classified_data[0]),'Classified data')
    print(len(data[0]),'Classified data ke nich ka')
    new_sigma=np.zeros((D,D,len(classified_data[0])))
    new_weight=np.zeros((len(classified_data[0]),1))
    sum_weight=0

    for j in range(len(classified_data[0])):
        sum_weight=sum_weight+np.sum(classified_data[:,j])
    print(sum_weight)
    for j in range(len(classified_data[0])):
        new_weight[j,0]=np.sum(classified_data[:,j])/sum_weight

    for k in range(D):
        for i in range(len(classified_data[0])):
            m=0
            for j in range(len(classified_data)):
                m = m+data[j,k]*classified_data[j,i]
            new_m[i,k]=m/np.sum(classified_data[:,i])

    for j in range(len(classified_data[0])):
        sigma_bb,sigma_rr,sigma_gg,sigma_bg,sigma_br,sigma_rg=0,0,0,0,0,0
        sum=np.sum(classified_data[:,j])
        for i in range(len(classified_data)):
            print(np.shape(data),'data')
            print(np.shape(new_m))
            print(np.shape(classified_data))
            print(np.sum(classified_data[:,j]))
            sigma_bb=sigma_bb+(classified_data[i,j]*(data[i,0]-new_m[j,0])**2)/sum
            sigma_gg=sigma_gg+(classified_data[i,j]*(data[i,1]-new_m[j,1])**2)/sum
            sigma_rr=sigma_rr+(classified_data[i,j]*(data[i,2]-new_m[j,2])**2)/sum
            sigma_bg=sigma_bg+(classified_data[i,j]*(data[i,0]-new_m[j,0])*(data[i,1]-new_m[j,1]))/sum
            sigma_br=sigma_br+(classified_data[i,j]*(data[i,0]-new_m[j,0])*(data[i,2]-new_m[j,2]))/sum
            sigma_rg=sigma_rg+(classified_data[i,j]*(data[i,2]-new_m[j,2])*(data[i,1]-new_m[j,1]))/sum
        print(sigma_rg)
        print(np.shape(new_sigma))
        new_sigma[:,:,j]=[[sigma_bb,sigma_bg,sigma_br],[sigma_bg,sigma_gg,sigma_rg],[sigma_br,sigma_rg,sigma_rr]]

    return new_weight,new_m,new_sigma

def data_generator(N):
    data1=[]
    data2=[]
    data3=[]

    total_hist=0
    for i in range(N+1):
        image=cv2.imread('Final%d.jpg' %i)
        for i in range (len(image)):
            for j in range (len(image[0])):
                if (int(image[i,j,0])>50 or int(image[i,j,1])>50 or int(image[i,j,1])>50) :
                    data1=np.append([data1],[int(image[i,j,0])])
                    data2=np.append([data2],[int(image[i,j,1])])
                    data3=np.append([data3],[int(image[i,j,2])])

    data1=np.array(data1)
    data2=np.array(data2)
    data3=np.array(data3)
    data=np.vstack((data1,data2,data3))
    data=data.transpose()

    return data

#main
data=data_generator(20)
#(k,D)
K=2
D=3
mean,sigma,weight=Initialize_parameter(K,D)
#print(mean)
mean=np.array([[50,240,200],[50,240,150]])
#print(mean)
sigma[:,:,0]=[[200,0,0],[0,10,0],[0,0,200]]
# print(LA.det(sigma[:,:,0]),'first')
# print(LA.eig(sigma[:,:,0]))
sigma[:,:,1]=[[10,0,0],[0,50,0],[0,0,60]]
# print(LA.det(sigma[:,:,1]),'sencod')
weight=np.array([[1/2],[1/2]])
check=-math.inf
while(True):
#for i in range(1):
    old_mean=mean
    old_sigma=sigma
    old_check=check
    #print(mean,sigma)
    classified_data,check=E_step(data,weight,mean,sigma,D)
    print(np.shape(data),'data')
    weight,mean,sigma=M_step(classified_data,data,D)
    print(weight,'weight')
    print(mean,'mean')
    print(sigma,'sigma')
    print(check,'checking condition')
    if (check<old_check):
        break
print(sigma)
print(mean)
