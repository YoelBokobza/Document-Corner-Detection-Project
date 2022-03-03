# Document-Corner-Detection-Project

This repository contains the implementation of my final project at the Introduction To Deep Learning Course, Electrical Engineering Departement, Ben Gurion University. 
Documents exist in both paper and digital form in our everyday life. Paper documents are easier to carry, read, and share whereas digital documents are easier to search, index, and store. For efficient knowledge management, it is often required to convert a paper document to digital format.   Document segmentation from the background is one of the major challenges of digitization from a natural image. An image of a document often contains the surface on which the document was placed to take the picture. It can also contain other non-document objects in the frame. Precise localization of document, often called document segmentation, from such an image is an essential first step required for digitization. Some of the challenges involved in localization are variations in document and background types, perspective distortions, shadows, and other objects in the image.

’ICDAR 2015 SmartDoc Challenge 1’ dataset is used for training and testing. The dataset contains 150 videos. Each video is around 10 seconds long and contains one of the six types of documents placed on one of the five distinct backgrounds. We split these videos into two disjoint sets.
Ground truth is available in the form of (x,y) coordinates labels:
top left (tl), top right (tr), bottom right (br), and bottom left (bl).

The short videos are taken over 5 different backgrounds, where background 5 is relatively complex as will be shown later.
Over each background, the videos have been taken over different document types - datasheet, letter, magazine, paper, patent, and tax, (5 short videos per background and document type).
# Data
The following google drive contains our trained networks checkpoints combined with the raw ICDAR data, in order to use this code and methods please use the data in the following link:
https://drive.google.com/drive/folders/1ZyWpjdwCUQZr6ZSNzQs1KpogdQX8atXU?usp=sharing

# Algorithms
This repository contains an implemenation of 2 different algorithms.

## First Solution - Combination of Hough Transform and Unet
We can divide this proposed algorithm into 4 main units:

### Preprocess
1.   Split the data into train, and test
2.   Resize the images to 512X512
3.   Normalize the dataset according to the train set mean and standard deviation.
4.   Apply a Gaussian filter over the images.

### Document Segmentation
Using the well-known deep learning architecture U-Net to create a binary mask segmentation of the document.

### Extracting Corners
The following steps details the algorithm we've developed to extract the 4 corners estimation coordinates from the masked image:
1. Extract edges from the masked image
2.   Apply Hough transform [2] over the edged masked image and extract some predefined number of peaks.
3.   Apply a hand-crafted clustering method to find an estimation of the 4 lines constructing the quadrilateral shape of the masked image
4. Extract corners from the lines equations

### Corners Estimation Refiner
We utilized the edged masked image to further improve the corners estimation.
![Hough](https://user-images.githubusercontent.com/49431639/156583140-7e9a53d3-7f7f-44a3-bdba-7c5f79663251.png)
