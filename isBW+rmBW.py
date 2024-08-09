import numpy as np
import cv2 
from retinaface import RetinaFace
import matplotlib.pyplot as plt
import os

"""
Part III (face filtering app)
Required features:
● Removing black and white images
● Face detection module
● Face feature extraction method (the features that you are about to use to determine
which the face is in front pose)
● Decision Function (a boolean function that returns true if the face is front pose)
● Filtering function (a function that inputs a directory and extract all front posed faces
from images and save them in another directory)
"""

def isBWorNot(img):
    threshold=30
    if len(img.shape) == 2:

        return True
    elif len(img.shape) == 3:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        back_to_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        diff = np.max(np.abs(img.astype(np.float32) - back_to_bgr.astype(np.float32)))
        return diff <= threshold
    
    return False

def rmvBWimg(imgpth):
    boolBW = isBWorNot(img)
    if boolBW == True:
        os.path.exists(imgpth)
        os.remove(imgpth)



img = cv2.imread('faces/testDATA.jpg')
imgpth = 'faces/testDATA.jpg'
isBW = isBWorNot(img)
rmvBWimg(imgpth)
