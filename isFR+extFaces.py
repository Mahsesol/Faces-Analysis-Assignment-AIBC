import cv2
from retinaface import RetinaFace
import numpy as np
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

def isFront(facelandmarks):
    Leye = np.array(facelandmarks['left_eye'])
    Reye = np.array(facelandmarks['right_eye'])
    nose = np.array(facelandmarks['nose'])
    eyeDstnc = np.linalg.norm(Reye - Leye)
    NELdistance = np.linalg.norm(nose - Leye)
    NERdistance = np.linalg.norm(nose - Reye)
    
    if abs(NELdistance - NERdistance) < 0.1 * eyeDstnc:
        return True
    else:
        return False

def extractFaces(inputDir, outputDir):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        
    for filename in os.listdir(inputDir):
        img_path = os.path.join(inputDir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = RetinaFace.detect_faces(img)
        if not faces:
            continue
        for i, (key, faceInfo) in enumerate(faces.items()):
            faceLandmarks = faceInfo['landmarks']
            if isFront(faceLandmarks):
                faceBox = faceInfo['facial_area']
                x, y, w, h = faceBox
                face_img = img[y:h, x:w]
                save_path = os.path.join(outputDir, f"{filename.split('.')[0]}_face{i}.jpg")
                cv2.imwrite(save_path, face_img)

extractFaces('faces', 'faces/extractedface')
