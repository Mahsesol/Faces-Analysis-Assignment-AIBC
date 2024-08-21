# Faces-Analysis-Assignment-AIBC
_A part of my AI summer camp journey!_

---

This project focuses on various image processing and computer vision tasks using OpenCV, RetinaFace. It includes modules for face detection, face pose analysis, and image filtering. The main goal is to filter out black-and-white images, detect faces, and extract front-facing faces from images in a directory.

## Contents
- [Installation](#installation)
- [Introduction](#introduction)
- [Usage](#usage)
- [Examples](#examples)

 
## Installation

### Prerequisites
Ensure you have Python 3.x installed. You also need to install the required Python libraries:

```bash
pip install numpy opencv-python retinaface matplotlib
```

### Clone the Repository
```bash
git clone https://github.com/Mahsesol/Faces-Analysis-Assignment-AIBC.git
cd Faces-Analysis-Assignment-AIBC
```

 
## Introduction

This project is a comprehensive face filtering and feature extraction tool that can:
- Remove black-and-white images.
- Detect faces in images.
- Determine whether a face is in a front-facing pose.
- Extract and save front-facing faces from a set of images.

The project consists of three main parts:
1. **Removing Black and White Images:** This module identifies and removes grayscale images from a directory.
2. **Face Detection and Pose Analysis:** Using RetinaFace for face detection, this module extracts facial landmarks to determine if the face is front-facing.
3. **Kernel Convolution on Images:** A custom convolution function that applies various kernels like Identity, Sobel, and Blur to images.

## Usage

### Part I: Removing Black and White Images
To remove black-and-white images from a directory:

```python
import cv2
import os
from isBW+rmBW import isBWorNot, rmvBWimg

img = cv2.imread('faces/6.jpeg')
imgpth = 'faces/6.jpeg'
isBW = isBWorNot(img)
rmvBWimg(imgpth)
```

### Part II: Face Detection and Pose Filtering
To detect faces and filter out non-front-facing ones:

```python
from isFR+extFaces import extractFaces

extractFaces('faces', 'faces/extractedface')
```

### Part III: Kernel Convolution
To apply different convolution kernels to an image:

```python
import cv2
import numpy as np
from kernels import krnlFunc

img = cv2.imread("faces/5.jpeg", cv2.IMREAD_GRAYSCALE)

identity_result = krnlFunc(img, identity_kernel)
# Display or save results as needed
```

## Examples

### Black and White Image Removal
Given a directory of images, this feature will remove any black-and-white images:

```python
# Example: Remove black and white images
rmvBWimg('faces/6.jpeg')
```

### Face Detection and Extraction
Extract front-facing faces from a directory of images:

```python
# Example: Extract front-facing faces
extractFaces('faces', 'faces/extractedface')
```

### Image Convolution
Apply a Sobel filter to an image:

```python
# Example: Apply left Sobel kernel
sobel_result = krnlFunc(img, left_sobel_kernel)
```


 
## Feedback

If you have any feedback, please reach out to me. <3 
## 
- [Mahsesol](https://www.github.com/Mahsesol)


