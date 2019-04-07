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

    for i in range(start,end+1):
        mat[i]=mat[i]*weight
        sum=sum+mat[i]
        Harsh=Harsh+i*mat[i]
    mean=Harsh +r

    for i in range(start,end+1):
        var_sum=var_sum+(i-mean)**2*mat[i]

    return sum,mean,np.sqrt(var_sum/(end-start))+r

def M_step(classified_data,data):
    r=0.0000001

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
        new_m[i,0]=m/np.sum(classified_data[:,i])+r
    #print(new_m,'new m')


    for j in range(len(classified_data[0])):
        sigma=0
        for i in range(len(classified_data)):
            difference=(data[i]-new_m[j,0])
            sigma=sigma+classified_data[i,j]*difference**2
        new_sigma[j,0]=np.sqrt(sigma/np.sum(classified_data[:,j]))+r
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

data=data_generator(20,2)
#print(len(data))
#print(data)

Initialize_parameter(3)
mean=[50,240,150]
sigma=[10,15,20]
weight=[1/3,1/3,1/3]

while(True):
#for i in range(5):
    old_mean=mean
    old_sigma=sigma
    #print(mean,sigma)
    classified_data=E_step(data,weight,mean,sigma)
    print(np.sum(classified_data[0,:]),'sum')
    print(np.sum(classified_data[50,:]),'sum')
    #print(np.shape(classified_data),'classified')
    #print(sum(classified_data[0,:]),sum(classified_data[10,:]))
    #print(classified_data[50,0],'class 50')
    #print(classified_data[50,1],'class 50')
    #print(mean,'old_mean')
    #print(sigma,'old_sigma')
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


'''
# -- coding: utf-8 --
"""
Created on Fri Apr  5 00:54:59 2019

@author: Aalap
"""
import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt

def Gaussian(x_, mean_, cov_):
    dim = mean_.shape[0]
    x_hat = (x_-mean_).reshape((-1,1,dim))
    N = np.sqrt((2*np.pi)**dim * det(cov_))
    fac = np.einsum('...k,kl,...l->...', x_hat, inv(cov_), x_hat)
    return np.exp(-fac / 2) / N

def generate_data(mean_, cov_, size_):
    return np.random.multivariate_normal(mean_, cov_, size_)

class GaussianMixture:

    input: data and number of clusters

    def _init(self, data, K_):
        self.data = data_
        self.dim = self.data.shape[1]  # dimension of the Guassian model
        self.K = K  # number of clusters

        # Initialize the mean
        num = self.data.shape[0]//self.K

        self.mean = []
        self.cov = []
        for k in range(self.K):
            self.mean.append(np.sum(self.data[k*num:(k+1)*num, ...], axis=0)/num)
            self.cov.append(np.identity(self.dim))
        self.weight = np.array([1.0 for k in range(self.K)])

    def expectation(self):
        prob_con = [Gaussian(self.data, self.mean[k], self.cov[k])*self.weight[k] for k in range(self.K)]
        total = sum(prob_con)  # sum up over clusters
        self.prob = prob_con/total

    def maximization(self):
        for k in range(self.K):
            p_total = np.sum(self.prob[k])
            p_weighted = self.prob[k].reshape((-1,1))*self.data

            #  update new mean
            p_weighted_sum = np.sum(p_weighted, axis=0)
            self.mean[k] = p_weighted_sum/p_total

            # update new covarience
            p_hat = self.data-self.mean[k]
            p_cov = self.prob[k].reshape((-1, 1, 1))*p_hat[:,:,None]*p_hat[:,None,:]  # batch cross dot
            p_cov = np.sum(p_cov, axis=0)
            self.cov[k] = p_cov/p_total
            self.weight[k] = p_total/self.data.shape[0]

    def train(self):
        self.expectation()
        self.maximization()

    def getModel(self):
        return (self.mean, self.cov, self.weight)

    def getPdf(self, x_):
        pdf = np.zeros(x_.shape[0])
        for k in range(self.K):
            print(Gaussian(x_, self.mean[k], self.cov[k]).shape)
            print(self.weight[k].shape)
            print((self.weight[k]*Gaussian(x_, self.mean[k], self.cov[k])).shape)
            print(pdf.shape)
            pdf += self.weight[k]*Gaussian(x_, self.mean[k], self.cov[k]).reshape((-1))
        return pdf

if _name_ == "_main_":
    #  Guassian model: [mean, cov]
    G1 = [np.array([0]), np.identity(1)*0.2]
    G2 = [np.array([1]), np.identity(1)*0.1]
    G_list = [G1, G2]
    K = len(G_list)  # number of clusters

    #  Generate training data
    data_list = [generate_data(G_list[k][0], G_list[k][1], 50) for k in range(K)]
    data = np.concatenate(data_list)

    #  Mixture Gaussian model
    n_iterations = 20
    mix = GaussianMixture(data, len(G_list))
    [mix.train() for i in range(n_iterations)]
    mean, var, _ = mix.getModel()

    #  Plot results

    plt.figure()
    plt.plot(data[:,0], data[:,1], '.')
    circle = []
    for k in range(K):
        circle.append(plt.Circle(mean[k], np.sqrt(var[k][0,0]), edgecolor='r', facecolor='none'))
        plt.gcf().gca().add_artist(circle[k])

    plt.show()

    #  Generate ground truth
    x_gt = np.linspace(-10, 15, 300)
    y_gt_list = [Gaussian(x_gt, G_list[k][0], G_list[k][1]) for k in range(K)]
    y_gt = sum(y_gt_list)/K

    plt.figure()
    plt.plot(x_gt, y_gt)

    #  Plot results
    data = np.sort(data, axis=0)
    prob = mix.getPdf(data)
    plt.plot(data, prob, '-.')
    plt.show()
'''
