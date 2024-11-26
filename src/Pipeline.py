import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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
    maxRed, maxGreen, maxBlue = 0,0,0
    minRed, minGreen, minBlue = 1,1,1
    
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
                
    #Iterate through image and normalize intensity values for each channel
    for y in range(inputHeight):
        for x in range(inputWidth):
            #Store pixel to avoid multiple accesses
            pixel = inputImg[y][x]
            
            #Calculate ranges for each color channel
            rangeRed = maxRed - minRed
            rangeGreen = maxGreen - minGreen
            rangeBlue = maxBlue - minBlue
            
            #Calculate new value for each channel
            newRed = (pixel[0] - minRed) / rangeRed
            newGreen = (pixel[1] - minGreen) / rangeGreen
            newBlue = (pixel[2] - minBlue) / rangeBlue
            
            #Save new values into image
            outputImg[y][x] = [newRed, newGreen, newBlue]
            
    return outputImg
            
#Function to reconstruct color channel from information from the green channel
#Primarily used for red
#Input: Three-channel image, channel to reconstruct, alpha value to use
#Output: Three-channel image with specified channel reconstructed
def reconstructChannel(inputImg, channel, alpha):
    #Extract image size information
    inputHeight, inputWidth = inputImg.shape[:2]
    
    #Calculate mean intensity for each channel
    meanRed = np.mean(inputImg[:,:,0])
    meanGreen = np.mean(inputImg[:,:,1])
    meanBlue = np.mean(inputImg[:,:,2])
    
    #Determine mean to use based on channel parameter
    meanChannel = 0.0
    if (channel == 0):
        meanChannel = meanRed
    else:
        meanChannel = meanBlue
    
    #Declare array to hold output image
    outputImg = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)
    
    for y in range(inputHeight):
        for x in range(inputWidth):
            #Store pixel value to avoid multiple array accesses
            pixel = inputImg[y][x]
            
            #Calculate reconstructed value using Ancuti's equation
            newVal = pixel[channel] + (alpha) * (meanGreen - meanChannel) * (1 - pixel[channel]) * pixel[1]

            #Store new value in output image
            pixel[channel] = newVal
            outputImg[y][x] = pixel
            
    #Return image with channel reconstructed
    return outputImg

#Function to white balance using Gray World algorithm
#Input: Three-channel image
#Output: White balanced three-channel image
def grayWorld(inputImg):
    #Calculate new image values by normalizing each channel for gray mean
    grayworldImg = (inputImg * (inputImg.mean() / inputImg.mean(axis=(0,1)))).clip(0, 1).astype(np.float32)
    
    #Retun balanced image
    return grayworldImg

#Function to produce sharpened image using unsharp masking in order to preserve details in fusion process
#Input: Three-channel image, 2D kernel size, and sigma
#Output: Sharpened three-channel image
def sharpenImage(inputImg, ksize, sigma):
    #Extract input size information
    inputHeight, inputWidth = inputImg.shape[:2]
    
    #Declare arrays to hold smoothed, convolved, and sharpened images
    gaussedImg = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)
    diffImg = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)
    sharpedImg = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)

    #Convolve input with Gaussian filter
    gaussedImg = cv.GaussianBlur(inputImg, ksize, sigma)
    
    #Subtract gaussed image from original, then normalize to fill entire dynamic range
    diffImg = normalizeChannelsFull(inputImg - gaussedImg)
    
    #Average input image with the results of the unsharp mask to produce sharpened image
    sharpedImg = (inputImg + diffImg) / 2

    #Display output
    fig, ax = plt.subplots()
    ax.imshow(sharpedImg)
    ax.plot()
    plt.show()

#Get paths for input and output files
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])

imgInput = plt.imread(inPath)
imgNormed = normalizeChannelsUpper(imgInput)
imgGrayNoRed = grayWorld(imgNormed)
imgRed = reconstructChannel(imgNormed, 0, 1)
imgGray = grayWorld(imgRed)
sharpenImage(imgGray, (5, 5), 1)



fig, ax = plt.subplots()
ax.imshow(imgGrayNoRed)
ax.plot()
plt.show()

img = plt.imsave(outPath, imgGray)