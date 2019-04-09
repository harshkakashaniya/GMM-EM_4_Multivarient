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


coordinates=np.zeros((8,2,140))
def onclick(event):
    if(event.xdata != None and event.ydata != None and event.button!=8 and click<8):
        print(event.xdata, event.ydata)
        global click
        print(click,"click")
        coordinates[click,0,image_count]=event.xdata
        coordinates[click,1,image_count]=event.ydata
        click+=1

for image_count in range(0,140):
    print(image_count,"count")
    click=0
    im = plt.imread('%d.jpg'%image_count)
    implot = plt.imshow(im)
    #implot = plt.imshow('image_count',im)
    #implot = cv2.imshow('%d'%image_count,im)

    while(click<=7):
        print(click,"click")
        cid = implot.figure.canvas.mpl_connect('button_press_event', onclick)
        if click==7:
            implot.figure.canvas.mpl_disconnect(cid)
        print(click,"otcd")
        plt.show()
        #implot.figure.canvas.mpl_disconnect(cid)

    #if click==8:
    #
