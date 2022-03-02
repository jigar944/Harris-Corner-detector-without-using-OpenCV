# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 16:12:51 2022

@author: jigar
"""

import cv2
import numpy as np


def HarrisCornerDetection(image1,img): 
      
    gaussian_blur_image = cv2.GaussianBlur(image1,(3,3),2)
       
    #sobel kernal x
    I_x = cv2.Sobel(gaussian_blur_image,cv2.CV_64F,1,0,ksize=3)
    
    #sobel kernal y
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
            
    cv2.namedWindow("R", cv2.WINDOW_NORMAL)
    cv2.imshow("R",final_image)
    #   Step 6 - Apply a threshold
    cv2.normalize(final_image, final_image, 0, 1, cv2.NORM_MINMAX)
    corners = []
    for y in range(10,height-9):
        for x in range(10,width-9):
                maxx = final_image[y,x]               
                neighbouhood = final_image[y-8:y+8,x-8:x+8]
                if np.amax(neighbouhood)<=maxx:
                    corners.append((y,x))
    
    
    #   Step 6 - Apply a threshold
    cv2.normalize(final_image, final_image, 0, 1, cv2.NORM_MINMAX)
    corners = []
    for y in range(10,height-9):
        for x in range(10,width-9):
                maxx = final_image[y,x]               
                neighbouhood = final_image[y-2:y+2,x-2:x+2]
                if np.amax(neighbouhood)<=maxx and maxx<0.9:
                    corners.append((y,x))
                    
    
    corners = sorted(corners,key=lambda x: final_image[x[0],x[1]],reverse=True)
    if len(corners)>500:
        corners = corners[:500]

    return final_image,corners 


main_img = cv2.imread('hough1.png')
image1 = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
       
a,c = HarrisCornerDetection(image1,main_img)
cv2.drawKeypoints(main_img,c, main_img, color=(255, 0, 255))

cv2.namedWindow("harris corner detected Image", cv2.WINDOW_NORMAL)
cv2.imshow("harris corner detected Image",main_img)
cv2.waitKey(0)