# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 16:12:51 2022

@author: jigar
"""



import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def create_gaussianfilter(size,sigma):
    gaussian_filter = np.zeros((size,size),np.float32)
    xx = yy = size//2
    for x in range(-xx,xx+1):
        for y in range(-yy,yy+1):
            step1 = 2*np.pi*(sigma**2)
            step2 = np.exp(-(x**2 + y**2)/(2*sigma**2))
            gaussian_filter[x+xx,y+yy] = (1/step1)*step2        
    return gaussian_filter

def convolution(image,kernal,kernal_sum):
    image_h = image.shape[0]
    image_w = image.shape[1]
    kernal_h = kernal.shape[0]
    kernal_w = kernal.shape[1]
    
    height = kernal_h//2
    width = kernal_w//2
    
    new_conv_image = np.zeros(shape=(image_h,image_w))
    
    for i in range(height,image_h-kernal_h):
        for j in range(width,image_w-kernal_w):
            x = image[i-1:i+kernal_h-1,j-1:j+kernal_w-1]
            m = np.multiply(kernal,x)
            new_conv_image[i][j] = np.absolute(np.sum(m)/kernal_sum)

    return new_conv_image

def HarrisCornerDetection(image1,img): 
      
    gaussian_kernal = create_gaussianfilter(3,2)
    y = np.sum(np.array(gaussian_kernal))
    gaussian_blur_image = convolution(image1, (gaussian_kernal),y)
    
    
    
    #sobel kernal x
    I_x = cv2.Sobel(gaussian_blur_image,cv2.CV_64F,1,0,ksize=3)
    
    
    #sobel kernal y
    sobel_kernal_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    I_y = cv2.Sobel(gaussian_blur_image,cv2.CV_64F,0,1,ksize=3)
    
    
    Ixx = np.square(I_x)
    cv2.namedWindow("Sobel_xx", cv2.WINDOW_NORMAL)
    cv2.imshow("Sobel_xx",Ixx/255)
    Iyy =np.square(I_y)
    cv2.namedWindow("Sobel_yy", cv2.WINDOW_NORMAL)
    cv2.imshow("Sobel_yy",Iyy/255)
    
    IxIy = np.multiply(I_x,I_y)
    cv2.namedWindow("IxIy", cv2.WINDOW_NORMAL)
    cv2.imshow("IxIy",IxIy/255)
    
    
    window_size=3
    height,width = image1.shape
    final_image = np.zeros((height,width))
    offset = int( window_size / 2 )
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            
            Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
            Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(IxIy[y-offset:y+1+offset, x-offset:x+1+offset])
    
            H = np.array([[Sxx,Sxy],[Sxy,Syy]])
    
            det=np.linalg.det(H)
            tr=np.matrix.trace(H)
            R=det-0.04*(tr**2)
            final_image[y-offset, x-offset]=R
    
    #   Step 6 - Apply a threshold
    cv2.normalize(final_image, final_image, 0, 1, cv2.NORM_MINMAX)
    corners = []
    for y in range(10,height-9):
        for x in range(10,width-9):
                maxx = final_image[y,x]               
                flag = False
                neighbouhood = final_image[y-8:y+8,x-8:x+8]
                if np.amax(neighbouhood)<=maxx:
                    corners.append((y,x))
    
    
    corners = sorted(corners,key=lambda x: final_image[x[0],x[1]],reverse=True)
    count=0
    for i,j in corners:
        count+=1
        if count>10:
            break
        cv2.circle(main_img,(j,i),3,(0,255,0))
                
    cv2.namedWindow("R", cv2.WINDOW_NORMAL)
    cv2.imshow("R",final_image)
    
    cv2.namedWindow("harris corner detected Image", cv2.WINDOW_NORMAL)
    cv2.imwrite("harris corner detected Image.jpg",main_img)
    
    return final_image,corners


main_img = cv2.imread('hough2.png')
image1 = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)

img1 = cv2.imread('hough2.png',cv2.IMREAD_GRAYSCALE)           
HarrisCornerDetection(image1,main_img)



cv2.waitKey(0)