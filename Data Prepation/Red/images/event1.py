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

'''
def draw_circle(event,x,y,flags,param):
	global mouseX,mouseY
	#mouseX=0
	#mouseY=0
	if event == cv2.EVENT_LBUTTONDBLCLK:
		#cv2.circle(img,(x,y),100,(255,0,0),-1)
		mouseX,mouseY = x,y

#img = np.zeros((512,512,3), np.uint8)
img=plt.imread('0.jpg')
cv2.namedWindow('image')
implot=plt.imshow('image',img)

def on_press(event):
    print('you pressed', event.button, event.xdata, event.ydata)

#cid = fig.canvas.mpl_connect('button_press_event', on_press)

'''

im = plt.imread('0.jpg')
implot = plt.imshow(im)

def onclick(event):
    if event.xdata != None and event.ydata != None:
        print(event.xdata, event.ydata)
cid = implot.figure.canvas.mpl_connect('button_press_event', onclick)

plt.show()


#while(1):
#	cid = img.canvas.mpl_connect('button_press_event', on_press)
	#cv2.setMouseCallback('image',draw_circle)
	#k = cv2.waitKey(20) & 0xFF
	#if k == 27:
	#	break
	#elif k == ord('a'):
	#	print(mouseX,mouseY)
