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
from numpy import linalg as LA
import math
from PIL import Image
import matplotlib.pyplot as plt
import random
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2


coordinates=np.zeros((2,2,201)) # Array to store the coordinates of the clicks

def onclick(event):
    if(event.xdata != None and event.ydata != None and event.button!=8 and click<2):
        print(event.xdata, event.ydata)
        global click
        print(click,"clickinside")
        coordinates[click,0,image_count]=event.xdata
        coordinates[click,1,image_count]=event.ydata
        click+=1

for i in range(0,200):
    img=cv2.imread('%d.jpg'%i)
    clahe = cv2.createCLAHE(clipLimit=1., tileGridSize=(1,1))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    cv2.imwrite('lab_images/lab%d.jpg'%i, img2)

#Cropping of the image frames
for image_count in range(0,200,2):
    black_image=np.zeros((480,640,3))
    print(image_count,"count")
    click=0
    im = plt.imread('lab_images/lab%d.jpg'%image_count)
    implot = plt.imshow(im)

    while(click<=1):
        print(click,"clickwhile")
        cid = implot.figure.canvas.mpl_connect('button_press_event', onclick)
        if click==2:
            implot.figure.canvas.mpl_disconnect(cid)
        print(click,"otcd")
        plt.show()

    #Plotting of circular crop using half planes
    for x in range(0,479):
        for y in range(0,639):
            CircleObstacle=math.pow(x-coordinates[0,1,image_count],2)+math.pow(y-coordinates[0,0,image_count],2)
            rad=np.sqrt(math.pow((coordinates[1,0,image_count]-coordinates[0,0,image_count]),2)+math.pow((coordinates[1,1,image_count]-coordinates[0,1,image_count]),2))
            print(rad)
            if CircleObstacle-math.pow(rad,2)<=0:
                black_image[x,y]=[255,255,255]
    cv2.imwrite('Yellow_new/Crop/new%d.jpg'%image_count,black_image)
    i2=cv2.imread('Yellow_new/Crop/new%d.jpg'%image_count)
    im=cv2.imread('lab_images/lab%d.jpg'%image_count)
    i=cv2.bitwise_and(im,i2)
    print("new shape",np.shape(i))
    cv2.imwrite('Yellow_new/Train/Final%d.jpg'%image_count,i)
