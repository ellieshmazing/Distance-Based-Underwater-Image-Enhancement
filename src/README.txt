Pipeline.py includes all of the functions used in this project. They are separated according to purpose, with higher level functions towards the bottom. 

All functions to run Ancuti's pipeline are handled by the wrapper "enhanceImage()". Calling it will apply every step, and save both the intermediate and final images. enhanceImage() currently uses the default parameters set for each function (i.e. the sigma value for the Gaussian filters). 

The pipeline can be run with a call in the form of:
python Pipeline.py [relative path to input image] [relative path to output folder]

The program will handle the creation of the output folder.



I did not create a wrapper for the Background Restoration process, but all functions are under the header "BACKGROUND RESTORATION FUNCTIONS." A cluster image is generated with generateClusters(), then the background mask and buffer zone created with maskDilate(), and the final merging handled by restoreBackground().

Thank you!!!