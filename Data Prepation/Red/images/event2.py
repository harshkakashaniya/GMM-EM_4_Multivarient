import os, sys
import numpy as np
import matplotlib as plt
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2


def onclick(event):
    """Deal with click events"""
    button = ['left','middle','right']
    toolbar = plt.get_current_fig_manager().toolbar
    if toolbar.mode!='':
        print("You clicked on something, but toolbar is in mode {:s}.".format(toolbar.mode))
    else:
        print("You {0}-clicked coords ({1},{2}) (pix ({3},{4}))".format(button[event.button+1],\
                                                                             event.xdata,\
                                                                             event.ydata,\
                                                                             event.x,\
                                                                             event.y))


if __name__ == "__main__":
    fig = plt.figure()
    plt.plot([1,2,3],[4,5,1],'ko-',picker=5)
    fig.canvas.mpl_connect('key_press_event',ontype)
    fig.canvas.mpl_connect('button_press_event',onclick)
    fig.canvas.mpl_connect('pick_event',onpick)
    plt.show()
