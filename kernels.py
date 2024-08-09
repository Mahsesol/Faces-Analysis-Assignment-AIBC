import numpy as np
import cv2

"""
Part O (Kernels)
Write the convolution function that inputs an image and a kernel then output the convolution product of them (implement the operation from scratch)
Test this kernels: -Identity
-left sobel
-blur
-random kernel
"""
def krnlFunc(img, krnl):
    krnl_H, krnl_w = krnl.shape
    img_H, img_W = img.shape[:2]
    
    output_height = img_H - krnl_H + 1
    output_width = img_W - krnl_w + 1
    prdct = np.zeros((output_height, output_width))

    pad_height = krnl_H // 2
    pad_width = krnl_w // 2
    paddedimg = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    prdct = np.zeros_like(img)

    for i in range(img_H):
        for j in range(img_W):
            prdct[i, j] = np.sum(paddedimg[i:i + krnl_H, j:j + krnl_H] * krnl)
    
    return prdct
img = cv2.imread("faces/5.jpeg", cv2.IMREAD_GRAYSCALE)

identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
left_sobel_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
blur_kernel = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
random_kernel = np.random.rand(3, 3)

identity_result = krnlFunc(img, identity_kernel)
sobel_result = krnlFunc(img, left_sobel_kernel)
blur_result = krnlFunc(img, blur_kernel)
random_result = krnlFunc(img, random_kernel)

cv2.imshow('Original Image', img)
cv2.imshow('Identity Kernel', identity_result)
cv2.imshow('Left Sobel Kernel', sobel_result)
cv2.imshow('Blur Kernel', blur_result)
cv2.imshow('Random Kernel', random_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
