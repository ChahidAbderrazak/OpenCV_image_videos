#%% -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:43:56 2019

@author: chahida
Email: abderrazak.chahid@gmail.com
"""
#% Packages and functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2

#from Python_lib.Shared_Functions import *

#%% ##################################################################################################
# input parameters

 #%%   ---------------------- OpenCV first code   -------------------------

Img = cv2.imread ("media/sami_yusuf.jpg",1)

cv2.imshow("Sami Yusuf", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 #%%   ---------------------- OpenCV first code   -------------------------
# Create a CascadeClassifier Object
face_cascade = cv2.CascadeClassifier("./Trained_Classifier/haarcascades/haarcascade_frontalface_default.xml")

# Reading the image as it is
img = cv2.imread("media/sami_yusuf.jpg")

# Reading the image as gray scale image
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#%% Search the co-ordintes of the image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05,
                                      minNeighbors=5)
for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),3)

cv2.imshow("Gray", img)
cv2.waitKey(0)
cv2.destroyAllWindows()




#%% Test Script CELL ################################################################################
print('The END.')

