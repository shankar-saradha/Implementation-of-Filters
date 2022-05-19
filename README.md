# Implementation-of-Filters
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import cv2, matplotlib.py libraries and read the saved images using cv2.imread().

### Step2
Convert the saved BGR image to RGB using cvtColor().

### Step3
By using the following filters for image smoothing:filter2D(src, ddepth, kernel), Box filter,Weighted Average filter,GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]), medianBlur(src, ksize),and for image sharpening:Laplacian Kernel,Laplacian Operator. 

### Step4
Apply the filters using cv2.filter2D() for each respective filters.

### Step5
Plot the images of the original one and the filtered one using plt.figure() and cv2.imshow().

## Program:
```
### Developed By   : SHANKAR SS
### Register Number: 212221240052
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("panda.jpeg")
original_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

## 1. Smoothing Filters
# i) Using Averaging Filter

kernel1 = np.ones((11,11),np.float32)/121
avg_filter = cv2.filter2D(original_image,-1,kernel1)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(avg_filter)
plt.title("Filtered")
plt.axis("off")

# ii) Using Weighted Averaging Filter

kernel2 = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
weighted_filter = cv2.filter2D(original_image,-1,kernel2)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(weighted_filter)
plt.title("Filtered")
plt.axis("off")

# iii) Using Gaussian Filter

gaussian_blur = cv2.GaussianBlur(src = original_image, ksize = (11,11), sigmaX=0, sigmaY=0)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Filtered")
plt.axis("off")

# iv) Using Median Filter

median = cv2.medianBlur(src=original_image,ksize = 11)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(median)
plt.title("Filtered (Median)")
plt.axis("off")

## 2. Sharpening Filters
# i) Using Laplacian Kernal

kernel3 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
laplacian_kernel = cv2.filter2D(original_image,-1,kernel3)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian_kernel)
plt.title("Filtered (Laplacian Kernel)")
plt.axis("off")

# ii) Using Laplacian Operator

laplacian_operator = cv2.Laplacian(original_image,cv2.CV_64F)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian_operator)
plt.title("Filtered (Laplacian Operator)")
plt.axis("off")
```

## OUTPUT:
### 1. Smoothing Filters
</br>

i) Using Averaging Filter
![download](https://user-images.githubusercontent.com/93978702/169313769-f000f3d7-86ce-47d8-a566-9f603e60ef21.png)


ii) Using Weighted Averaging Filter
![download](https://user-images.githubusercontent.com/93978702/169313833-72f66ff4-0c0e-4867-8242-6c55f9c8aa87.png)


iii) Using Gaussian Filter
![download](https://user-images.githubusercontent.com/93978702/169313886-6cb6c849-c29a-4ada-aeef-f928253c03d4.png)


iv) Using Median Filter
![download](https://user-images.githubusercontent.com/93978702/169313933-ac83d9b5-e9f7-4fc7-9dc1-ad0f0f11b09d.png)


### 2. Sharpening Filters
</br>

i) Using Laplacian Kernal
![download](https://user-images.githubusercontent.com/93978702/169313971-cbff3600-bb3b-4d68-a7f3-06bf76801957.png)



ii) Using Laplacian Operator
![download](https://user-images.githubusercontent.com/93978702/169314092-72938ac3-0e1c-41f8-ae74-df93cda6ff22.png)



## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
