import os
import sys
import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt
#from Pipeline import normalizeSingleChannelFull, normalizeChannelsFull
from skimage import color, filters, transform

def normalizeSingleChannelFull(inputImg):
    #Extract image size information
    inputHeight, inputWidth = inputImg.shape[:2]
    
    #Declare variables to hold max and min of each channel
    max = -10000
    min = 10000
    
    #Iterate through image to find max
    for y in range(inputHeight):
        for x in range(inputWidth):
            #Store pixel to avoid multiple array accesses
            pixel = inputImg[y][x]
        
            #Check if pixel intensity is greater than max
            if (pixel > max):
                max = pixel
                
            #Check if pixel intensity is less than min
            if (pixel < min):
                min = pixel
                    
    #Declare array to hold output image
    outputImg = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    
    #Calculate range
    rangePix = max-min
                
    #Iterate through image and normalize intensity values
    for y in range(inputHeight):
        for x in range(inputWidth):
            #Store pixel to avoid multiple accesses
            pixel = inputImg[y][x]

            #Calculate new value for each channel
            newPix = (pixel - min) / rangePix
            
            #Save new values into image
            outputImg[y][x] = newPix
            
    return outputImg

#Function to normalize color channels between [0, 1] based on upper limit of dynamic range
#Input: Three-channel image
#Output: Normalized three-channel image
def normalizeChannelsUpper(inputImg):
    #Extract image size information
    inputHeight, inputWidth = inputImg.shape[:2]
    
    #Declare variables to hold max of each channel
    maxRed, maxGreen, maxBlue = 0,0,0
    
    #Iterate through image to find max
    for y in range(inputHeight):
        for x in range(inputWidth):
            #Store pixel to avoid multiple array accesses
            pixel = inputImg[y][x]
        
            #Check if pixel intensity is greater than max for each channel
            if (pixel[0] > maxRed):
                maxRed = pixel[0]
            if (pixel[1] > maxGreen):
                maxGreen = pixel[1]
            if (pixel[2] > maxBlue):
                maxBlue = pixel[2]
                
    #Declare array to hold output image
    outputImg = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)
                
    #Iterate through image and normalize intensity values for each channel
    for y in range(inputHeight):
        for x in range(inputWidth):
            #Store pixel to avoid multiple accesses
            pixel = inputImg[y][x]
            
            #Calculate new value for each channel
            newRed = pixel[0] / maxRed
            newGreen = pixel[1] / maxGreen
            newBlue = pixel[2] / maxBlue
            
            #Save new values into image
            outputImg[y][x] = [newRed, newGreen, newBlue]
            
    return outputImg

#Function to normalize color channels between [0, 1] to utilize full dyanmic range
#Input: Three-channel image
#Output: Normalized three-channel image
def normalizeChannelsFull(inputImg):
    #Extract image size information
    inputHeight, inputWidth = inputImg.shape[:2]
    
    #Declare variables to hold max and min of each channel
    maxRed, maxGreen, maxBlue = -100,-100,-100
    minRed, minGreen, minBlue = 100,100,100
    
    #Iterate through image to find max
    for y in range(inputHeight):
        for x in range(inputWidth):
            #Store pixel to avoid multiple array accesses
            pixel = inputImg[y][x]
        
            #Check if pixel intensity is greater than max for each channel
            if (pixel[0] > maxRed):
                maxRed = pixel[0]
            if (pixel[1] > maxGreen):
                maxGreen = pixel[1]
            if (pixel[2] > maxBlue):
                maxBlue = pixel[2]
                
            if (pixel[0] < minRed):
                minRed = pixel[0]
            if (pixel[1] < minGreen):
                minGreen = pixel[1]
            if (pixel[2] < minBlue):
                minBlue = pixel[2]
                
    #Declare array to hold output image
    outputImg = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)
    
    #Calculate ranges for each color channel
    rangeRed = maxRed - minRed
    rangeGreen = maxGreen - minGreen
    rangeBlue = maxBlue - minBlue
                
    #Iterate through image and normalize intensity values for each channel
    for y in range(inputHeight):
        for x in range(inputWidth):
            #Store pixel to avoid multiple accesses
            pixel = inputImg[y][x]
            
            #Calculate new value for each channel
            newRed = (pixel[0] - minRed) / rangeRed
            newGreen = (pixel[1] - minGreen) / rangeGreen
            newBlue = (pixel[2] - minBlue) / rangeBlue
            
            #Save new values into image
            outputImg[y][x] = [newRed, newGreen, newBlue]
            
    return outputImg

#Function to perform K-Means clustering on weight maps to detect and segment background pixels
#Input: Laplacian weight maps of Gamma-Corrected and Sharpened images
#Output: Binary mask of image background
def maskBackground(inputImg, GammaLap, GammaSal, GammaSat, SharpLap, SharpSal, SharpSat, k = 3):
    #Extract image size
    imgHeight, imgWidth = inputImg.shape[:2]
    
    #Create array to hold variables for clustering
    pixelData = np.zeros(((imgHeight * imgWidth), 13), dtype=np.float32)
    
    #Create array to hold variables for clustering
    pixelData = np.zeros(((imgHeight * imgWidth), 6), dtype=np.float32)
    
    #Iterate through image and append values to each pixel
    pixelCount = 0
    for y in range(imgHeight):
        for x in range(imgWidth):
            pixelData[pixelCount][0] = GammaLap[y][x]
            pixelData[pixelCount][1] = SharpLap[y][x]
            pixelData[pixelCount][2] = SharpSal[y][x]
            pixelData[pixelCount][3] = GammaSal[y][x]
            pixelData[pixelCount][4] = 15 * (inputImg[y][x][0] - 2 / 255)
            pixelData[pixelCount][5] = y


            
            #Increment pixel
            pixelCount += 1
            
    #Declare parameters for cv.kmeans
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100000, .000001)
    
    #Execute K-means clustering
    returnData, labels, centers = cv.kmeans(pixelData, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    #Reshape labels to image dimensions
    clusteredImg = labels.reshape((imgHeight, imgWidth))
    
    #Normalize segmented image
    clusteredImgNorm = normalizeSingleChannelFull(clusteredImg)
    
    #Return result
    return clusteredImgNorm

def maskDilate(clusters, bgCluster):
    imgHeight, imgWidth = clusters.shape[:2]
    
    bg = np.zeros((imgHeight, imgWidth), dtype=np.uint8)
    bgDilated = np.zeros((imgHeight, imgWidth), dtype=int)
    
    #Add pixels to restored image based on cluster assignment
    for y in range(imgHeight):
        for x in range(imgWidth):
            if (np.mean(clusters[y][x]) == bgCluster):
                bg[y][x] = 1
            else:
                bg[y][x] = 0
               
    bgDilated = cv.dilate(bg, cv.getStructuringElement(cv.MORPH_RECT, (5,5)))
    bgDilatedTrans = cv.distanceTransform(bgDilated, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    
    for y in range(imgHeight):
        for x in range(imgWidth):
            if (bg[y][x] == 1):
                bgDilatedTrans[y][x] = 0
    
    return bg, bgDilatedTrans

#Function to restore the background of the image to its original color scheme
#Input: Enhanced image, original image, background mask, and dilated background mask
#Output: Enhanced image with restored background
def restoreBackground(inputImg, enhancedImg, bg, bgDilated):
    #Extract image size
    imgHeight, imgWidth = inputImg.shape[:2]
    
    #Create array to hold restored image
    bgRestoredImg = np.zeros((imgHeight, imgWidth, 4), dtype=np.float32)
    
    #Add pixels to restored image based on cluster assignment
    for y in range(imgHeight):
        for x in range(imgWidth):
            if (bgDilated[y][x][0] != 0):
                bgRestoredImg[y][x] = (inputImg[y][x] * bgDilated[y][x]) + (enhancedImg[y][x] * (1 - bgDilated[y][x]))
            elif (bg[y][x][0] == 1):
                bgRestoredImg[y][x] = inputImg[y][x]
            else:
                bgRestoredImg[y][x] = enhancedImg[y][x]
                
    #Return image with background restored
    return bgRestoredImg


srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
clusterVal = float(sys.argv[2])

imgInput = plt.imread(inPath + "Input.png")
enhancedImg = plt.imread(inPath + "Enhanced.png")
clusterImg = plt.imread(inPath + "Clusters.png")
bg = plt.imread(inPath + "map1.png")
bgD = plt.imread(inPath + "dist2.png")


bgR = restoreBackground(imgInput, enhancedImg, bg, bgD)

plt.imsave(inPath + 'final5.png', bgR)