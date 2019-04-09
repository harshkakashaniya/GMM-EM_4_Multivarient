import argparse
import numpy as np
import os, sys
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

A=cv2.imread('ab.png')
B=255*np.ones((200,200,3),dtype=np.uint8)
B[:,:,0]=int(0)
B[:,:,2]=int(0)
print(B[2,2,1],'zero')
#cv2.imshow('frame',B)
#cv2.waitKey()
#plt.hist(A[:,:,0])
#hist = cv2.calcHist([A[:,:,0]],[0],None,[255],[0,256])
#hist1 = cv2.calcHist([A[:,:,1]],[0],None,[255],[0,256])
hist2 = cv2.calcHist([A[:,:,2]],[0],None,[255],[0,256])
Harsh=0
'''
for i in range(len(hist2)):
   Harsh=Harsh+i*hist2[i,0]

Harsh=int(Harsh/np.sum(hist2))


ntst=0
for j in range(len(hist2)):
   ntst=ntst+(hist2[j,0]-hist2[Harsh,0])**2

print(np.sqrt(ntst/len(hist2)))
'''
for i in range(len(hist2)):
   Harsh=Harsh+hist2[i,0]

Harsh=int(Harsh/len(hist2))
ntst=0
for j in range(len(hist2)):
   ntst=ntst+(hist2[j,0]-Harsh)**2

#np.sqrt(ntst/len(hist2)-1)



mu = Harsh
variance = ntst/(len(hist2)-1)

#hstd = np.std(hist2)

#print(hstd)

sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))

#plt.hist(A[:,:,0],20)
#print(np.argmax(hist1))
#print(hist1[254,0])
#plt.plot(hist,'-b',label="Blue")
#plt.plot(hist1,'-g',label="Green")
#plt.plot(hist2,'-r',label="Red")

#plt.plot(hist1)
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.legend()
plt.ylabel("Frequency")
plt.show()
