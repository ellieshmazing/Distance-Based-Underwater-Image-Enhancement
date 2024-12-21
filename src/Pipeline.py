import os
import sys
import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, filters, transform

'''***********************************NORMALIZATION FUNCTIONS****************************************'''
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

'''***************************************NORMALIZATION FUNCTIONS***************************************'''

'''***************************************WHITE BALANCING FUNCTIONS**************************************'''
       
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

'''***************************************************WHITE BALANCING FUNCTIONS************************************************'''

'''***********************************************BACKGROUND RESTORATION FUNCTIONS*********************************************'''
#Function to perform K-Means clustering on weight maps to detect and segment background pixels
#Input: Laplacian weight maps of Gamma-Corrected and Sharpened images
#Output: Binary mask of image background
def generateClusters(inputImg, GammaLap, GammaSal, GammaSat, SharpLap, SharpSal, SharpSat, k = 3):
    #Extract image size
    imgHeight, imgWidth = inputImg.shape[:2]
    
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

#Function to generate background mask and a dilated version
#Input: Result of K-means clustering and background cluster value
#Output: Background mask and dilated background mask with distance transform
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

'''***********************************************BACKGROUND RESTORATION FUNCTIONS*********************************************'''

'''***************************************************FUSION INPUT FUNCTIONS***************************************************'''
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

    #Return the sharpened image
    return sharpedImg
    
#Function to perform gamma correction on given image
#Input: Three-channel image and gamma value
#Output: Gamma-corrected three-channel image
def gammaCorrection(inputImg, gamma):
    #Extract input size information
    inputHeight, inputWidth = inputImg.shape[:2]
    
    #Declare array to hold gamma-corrected image
    gammedImg = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)
    
    #Perform gamma correction
    gammedImg = inputImg ** gamma
    
    #Return gamma-corrected image
    return gammedImg

'''***************************************************FUSION INPUT FUNCTIONS***************************************************'''

'''***************************************************FUSION WEIGHTS FUNCTIONS*************************************************'''

#Function to determine saliency weights of each pixel in image
#Input: Three-channel image and sigma value for Gaussian filter
#Output: Array of saliency values for every pixel arranged in shape of input image
def saliencyWeights(inputImg, sigma):
    #Extract input size information
    inputHeight, inputWidth = inputImg.shape[:2]
    
    #Declare arrays to hold gaussed image, distance from mean, and saliency weights
    gaussedImgLab = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)
    meanDiffLab = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)
    saledImg = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)
    saledImgLab = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    saledImgNorm = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    saledImgGray = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    
    #Convert input image to Lab color space
    inputImgLab = color.rgb2lab(inputImg)
    
    #Determine mean vector by taking mean of each attribute
    inputImgLabMean = np.zeros(3, dtype=np.float32)
    inputImgLabMean[0] = np.mean(inputImgLab[:,:,0])
    inputImgLabMean[1] = np.mean(inputImgLab[:,:,1])
    inputImgLabMean[2] = np.mean(inputImgLab[:,:,2])
    
    #Convolve input image with Gaussian filter
    gaussedImgLab = cv.GaussianBlur(inputImg, (5, 5), sigma)
    
    #Subtract gaussedImg from image mean
    meanDiffLab = inputImgLabMean - gaussedImgLab
    
    #Calculate L2 norm for each pixel
    saledImgLab = (meanDiffLab ** 2) ** (1/2)
    
    #Convert image back to RGB colorspace
    saledImg = color.lab2rgb(saledImgLab)
    
    #Convert image to grayscale
    saledImgGray = color.rgb2gray(saledImg)
    
    #Normalize image channels
    saledImgNorm = normalizeSingleChannelFull(saledImgGray)
    
    #Return saliency values
    return saledImgNorm
    
#Function to determine the saturation weight for each pixel
#Input: Three-channel image
#Output: Array of weights for each pixel by amount of saturation
def saturationWeights(inputImg):
    #Extract input size information
    inputHeight, inputWidth = inputImg.shape[:2]
    
    #Declare arrays to hold luminance and saturation weights
    lumImg = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    satImg = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    satImgNorm = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    
    #Calculate luminance of every pixel
    lumImg = .2126 * inputImg[:,:,0] + .7152 * inputImg[:,:,1] + .0722 * inputImg[:,:,2]

    #Calculate saturation weights of every pixel
    satImg = ((1 / 3) * ((inputImg[:,:,0] - lumImg[:,:]) ** 2 + (inputImg[:,:,1] - lumImg[:,:]) ** 2 + (inputImg[:,:,2] - lumImg[:,:]) ** 2)) ** .5

    #Normalize saturation weights between [0,1]
    satImgNorm = normalizeSingleChannelFull(satImg)
    
    #Return weight map
    return satImgNorm

#Function to obtain Laplacian contrast weight map by finding the absolute value of the luminance channel passed through a Laplacian filter
#Input: 3-channel image and the low sigma for the DoG (high is 1.6 * low)
#Output: Single-channel Laplacian contrast weight map
def lapContrastWeights(inputImg, sigma):
    #Extract input size information
    inputHeight, inputWidth = inputImg.shape[:2]
    
    #Declare arrays to hold luminance, and the raw and normalized versions of the Laplacian contrast weights
    lumImg = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    lapImg = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    lapImgNorm = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    
    #Calculate luminance of every pixel
    lumImg = .2126 * inputImg[:,:,0] + .7152 * inputImg[:,:,1] + .0722 * inputImg[:,:,2]
    
    #Apply DoG filter to estimate Laplacian
    lapImg = filters.difference_of_gaussians(lumImg, sigma)
    
    #Normalize Laplacian contrast weights
    lapImgNorm = normalizeSingleChannelFull(lapImg)
    
    #Return normalized Laplacian weights
    return lapImgNorm

#Function to merge weight maps
#Input: Single-channel saturation, saliency, and Laplacian contrast weight maps, and delta normalization factor
#Output: Single-channel merged weight map
def mergeWeights(satImg, saledImg, lapImg, delta):
    #Extract input size information
    inputHeight, inputWidth = satImg.shape[:2]

    #Declare arrays to hold luminance, and the raw and normalized versions of the Laplacian contrast weights
    summedWeights = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    mergedWeights = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    
    #Sum weights
    summedWeights = satImg + saledImg + lapImg
    
    #Normalize weights on a pixel-by-pixel basis
    mergedWeights = (summedWeights + delta) / (satImg + saledImg + lapImg + (3 * (3 + delta)))
    
    #Return merged weight map
    return mergedWeights

'''***************************************************FUSION WEIGHTS FUNCTIONS*************************************************'''

'''***************************************************MULTI-SCALE FUSION FUNCTIONS********************************************************'''

#Function to compose Gaussian pyramid with maximum possible layers from input image
#Input: Weight map, sigma for Gaussian filter
#Output: Gaussian pyramid of downscaled versions of weight map
def generateGaussianPyramid(inputImg, sigma = (2/3)):
    #Generate Gaussian pyramid with maximum layers
    gausPyramid = tuple(transform.pyramid_gaussian(inputImg, max_layer = -1, downscale = 2, sigma = sigma))

    #Return both pyramids
    return gausPyramid

#Function to compose Laplacian pyramid with maximum possible layers from input image
#Input: Three-channel image, sigma for filter
#Output: Laplacian pyramid of downscaled versions of image
def generateLaplacianPyramid(inputImg, sigma = (2/3)):
    #Generate Laplacian pyramid with maximum layers
    lapPyramid = tuple(transform.pyramid_laplacian(inputImg, max_layer = -1, downscale = 2, sigma = sigma, channel_axis = 2))

    #Return both pyramids
    return lapPyramid

#Function to perform multi-scale fusion on the given pyramids
#Input: Gaussian pyramids for the weight maps, and Laplacian pyramids for the input images
#Output: Dehazed image
def fusePyramids(gausPyramidSharp, lapPyramidSharp, gausPyramidGamma, lapPyramidGamma):
    #Extract size of largest image in pyramid
    inputHeight, inputWidth = gausPyramidSharp[0].shape[:2]
    
    #Extract number of layers in pyramid
    pyramidLayers = len(gausPyramidSharp)
    
    #Declare array to hold the three-channel fused output for each pyramid layer
    fusedPyramidLayers = np.zeros((inputHeight, inputWidth, 3, pyramidLayers), dtype=np.float32)
    
    #Multiply the Gaussian and Laplacian layers for each input at each layer, then sum and upscale to original image size
    for i in range(pyramidLayers):
        #Declare array to hold the three-channel fused output for each pyramid layer
        sharpCombined = np.zeros((math.ceil(inputHeight / (2 ** i)), math.ceil(inputWidth / (2 ** i)), 3), dtype=np.float32)
        gammaCombined = np.zeros((math.ceil(inputHeight / (2 ** i)), math.ceil(inputWidth / (2 ** i)), 3), dtype=np.float32)
        
        #Multiply weight map Gaussian and input Laplacian for each input for each color channel
        sharpCombined[:,:,0] = gausPyramidSharp[i] * lapPyramidSharp[i][:,:,0]
        sharpCombined[:,:,1] = gausPyramidSharp[i] * lapPyramidSharp[i][:,:,1]
        sharpCombined[:,:,2] = gausPyramidSharp[i] * lapPyramidSharp[i][:,:,2]
        
        gammaCombined[:,:,0] = gausPyramidGamma[i] * lapPyramidGamma[i][:,:,0]
        gammaCombined[:,:,1] = gausPyramidGamma[i] * lapPyramidGamma[i][:,:,1]
        gammaCombined[:,:,2] = gausPyramidGamma[i] * lapPyramidGamma[i][:,:,2]
        
        #Sum combined values and resize to original image shape
        fusedPyramidLayers[:,:,:,i] = transform.resize((sharpCombined + gammaCombined), (inputHeight, inputWidth, 3))
    
    #Return fused pyramid layers
    return fusedPyramidLayers

#Function to sum all layers of the fused pyramid
#Input: Fused pyramid
#Output: Dehazed image
def mergePyramidLayers(fusedPyramidLayers):
    #Extract size of largest image in pyramid
    inputHeight, inputWidth = fusedPyramidLayers[:,:,:,0].shape[:2]
    
    #Extract number of layers in pyramid
    numPyramidLayers = len(fusedPyramidLayers[0,0,0])
    
    #Declare array to hold final merged image
    mergedImg = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)
    
    #Iterate through pyramid layers and calculate sum
    for i in range(numPyramidLayers):
        mergedImg += fusedPyramidLayers[:,:,:,i]
        
    #Return merged layers
    return mergedImg
    
'''***************************************************MULTI-SCALE FUSION FUNCTIONS********************************************************'''

'''***************************************************PIPELINE ORCHESTRATION FUNCTIONS*****************************************'''

#Function to perform white balancing process with red reconstruction
#Input: Three-channel image and alpha value for red reconstruction
#Output: White-balanced three-channel image
def whiteBalance(inputImg, channel = 0, alpha = 1):
    #Extract input size information
    inputHeight, inputWidth = inputImg.shape[:2]

    #Declare arrays to hold normalized, red-reconstructed, and post-gray-world images
    inputImgNorm = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)
    inputImgRedRec = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)
    inputImgGrayWorld = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)
    
    #Normalize pixel values between [0,1] based on upper limit of dynamic range
    inputImgNorm = normalizeChannelsUpper(inputImg)
    
    #Reconstruct red channel from green values
    inputImgRedRec = reconstructChannel(inputImgNorm, channel, alpha)
    
    #Apply gray world algorithm for white balancing
    inputImgGrayWorld = grayWorld(inputImgRedRec)
    
    #Return white-balanced image
    return inputImgGrayWorld, inputImgRedRec

#Function to generate inputs for fusion stage
#Input: White-balanced three-channel image
#Output: Gamma-corrected three-channel image and sharpened three-channel image
def generateFusionInputs(inputImg, ksize = (5,5), sigma = 4, gamma = 2):
    #Extract input size information
    inputHeight, inputWidth = inputImg.shape[:2]

    #Declare arrays to hold normalized, red-reconstructed, and post-gray-world images
    inputGamma = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)
    inputSharped = np.zeros((inputHeight, inputWidth, 3), dtype=np.float32)
    
    #Sharpen input image
    inputSharped = sharpenImage(inputImg, ksize, sigma)
    
    #Gamma-correct input image
    inputGamma = gammaCorrection(inputImg, gamma)
    
    #Return results
    return inputSharped, inputGamma

#Function to generate merged weight map
#Input: Three-channel image
#Output: Merged saliency, saturation, and Laplacian contrast weight map
def generateMergedWeightMap(inputImg, sigmaSal = 1, sigmaLap = 2, delta = .1, save = False, version = 'Sharp', outPath = None):
    #Extract input size information
    inputHeight, inputWidth = inputImg.shape[:2]

    #Declare arrays to hold saliency, saturation, Laplacian contrast, and merged weight maps
    weightSat = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    weightSal = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    weightLap = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    weightMerged = np.zeros((inputHeight, inputWidth), dtype=np.float32)
    
    #Generate each weight map
    weightSat = saturationWeights(inputImg)
    weightSal = saliencyWeights(inputImg, sigmaSal)
    weightLap = lapContrastWeights(inputImg, sigmaLap)
    
    #Save intermediate images if desired
    if (save):
        plt.imsave(outPath + version + 'SatMap.png', weightSat, cmap = 'gray')
        plt.imsave(outPath + version + 'SalMap.png', weightSal, cmap = 'gray')
        plt.imsave(outPath + version + 'LapMap.png', weightLap, cmap = 'gray')
    
    #Merge weight maps
    weightMerged = mergeWeights(weightSat, weightSal, weightLap, delta)
    
    #Return merged weight map
    return weightMerged, weightSal, weightLap, weightSat

#Function to perform multi-scale fusion of the inputs and their weight maps
#Input: Inputs and weight maps
#Output: Dehazed, fused image
def multiscaleFusion(inputSharp, weightSharp, inputGamma, weightGamma):
    #Generate Gaussian pyramids for each weight map
    gausPyramidSharp = generateGaussianPyramid(weightSharp)
    gausPyramidGamma = generateGaussianPyramid(weightGamma)
    
    #Generate Laplacian pyramids for each input
    lapPyramidSharp = generateLaplacianPyramid(inputSharp)
    lapPyramidGamma = generateLaplacianPyramid(inputGamma)
    
    #Combine each pyramid layer
    fusedPyramids = fusePyramids(gausPyramidSharp, lapPyramidSharp, gausPyramidGamma, lapPyramidGamma)
    
    #Perform multi-scale fusion on pyramid layers
    fusedImg = mergePyramidLayers(fusedPyramids)
    
    #Return fused (and therefore dehazed) image
    return fusedImg

#Function to enhance image by running it through entire pipeline
#Input: Input image and whether to save intermediate results
#Output: White-balanced and dehazed image
def enhanceImage(inputImg, save = False, outPath = None, k = 3):
    #Make directory to save output images
    if (save):
        os.mkdir(outPath)
                
    #White-balance input image
    imgGrayWorld, imgRedReconstructed = whiteBalance(inputImg)
    
    #Generate fusion inputs
    inputSharped, inputGamma = generateFusionInputs(imgGrayWorld)

    #Generate merged weight maps for each input
    weightSharp, sharpSal, sharpLap, sharpSat = generateMergedWeightMap(inputSharped, save = save, version = 'Sharp', outPath = outPath)
    weightGamma, gammaSal, gammaLap, gammaSat = generateMergedWeightMap(inputGamma, save = save, version = 'Gamma', outPath = outPath)

    #Extract background
    clusteredImg = generateClusters(inputImg, gammaLap, gammaSal, gammaSat, sharpLap, sharpSal, sharpSat, k)

    #Fuse inputs and their weight maps
    fusedImg = multiscaleFusion(inputSharped, weightSharp, inputGamma, weightGamma)

    #Normalize fused image
    resultImg = normalizeChannelsFull(fusedImg)
    
    #Save intermediate steps if desired
    if (save):
        plt.imsave(outPath + 'Input.png', inputImg)
        plt.imsave(outPath + 'RedReconstructed.png', imgRedReconstructed)
        plt.imsave(outPath + 'GrayWorld.png', imgGrayWorld)
        plt.imsave(outPath + 'Sharp.png', inputSharped)
        plt.imsave(outPath + 'SharpMergedMap.png', weightSharp, cmap = 'gray')
        plt.imsave(outPath + 'Gamma.png', inputGamma)
        plt.imsave(outPath + 'GammaMergedMap.png', weightGamma, cmap = 'gray')
        plt.imsave(outPath + 'Clusters.png', clusteredImg, cmap='gray')
        plt.imsave(outPath + 'Enhanced.png', resultImg)
        
    #Return final image
    return resultImg

'''***************************************************PIPELINE ORCHESTRATION FUNCTIONS*******************************************'''

#Get paths for input and output files
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])

imgInput = plt.imread(inPath)

result = enhanceImage(imgInput, True, outPath, 3)