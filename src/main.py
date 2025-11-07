import numpy as np
import pandas as pd
import cv2
import os
from skimage.filters import sobel, prewitt, scharr, roberts
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, '..', 'Datasets', 'DRIVE', 'training', 'images', '21_training.tif')

image = cv2.imread(image_path, cv2.IMREAD_COLOR)

if image is None:
    print("Error: Could not read the image.")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Image", image)
cv2.imshow("Gray Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

df = pd.DataFrame()
#print("Image DataFrame:")
#print(df)
image2 = np.reshape(gray, -1)
df['pixel_value'] = image2

# Gabor filters
num = 1
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for lambd in np.arange(3, 9, 2):  # [3, 5, 7]:
            for gamma in (0.05, 0.5):
                gabor_label = 'Gabor' + str(num)
                ksize = 5
                kernel = gabor = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)
                fimage = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                filtered_image2 = np.reshape(fimage, -1)
                df[gabor_label] = filtered_image2
                print(  f'Gabor_label: {gabor_label}, theta = {theta}, sigma = {sigma}, lambda = {lambd}, gamma = {gamma}' )
                num += 1

# Canny edge detector
edge = cv2.Canny(gray, 35, 200)

cv2.imshow("Gray Image for Canny", gray)
cv2.imshow("Canny Edge Detection", edge)
cv2.waitKey(0)
cv2.destroyAllWindows()

edge2 = np.reshape(edge, -1)

df['Canny_edge'] = edge2

edge_sobel = sobel(gray)
edge_sobel2 = np.reshape(edge_sobel, -1)
df['Sobel_edge'] = edge_sobel2

edge_prewitt = prewitt(gray)
edge_prewitt2 = np.reshape(edge_prewitt, -1)
df['Prewitt_edge'] = edge_prewitt2

edge_scharr = scharr(gray)
edge_scharr2 = np.reshape(edge_scharr, -1)
df['Scharr_edge'] = edge_scharr2

edge_roberts = roberts(gray)
edge_roberts2 = np.reshape(edge_roberts, -1)
df['Roberts_edge'] = edge_roberts2

cv2.imshow("Sobel Edge Detection", edge_sobel)
cv2.imshow("Prewitt Edge Detection", edge_prewitt)
cv2.imshow("Scharr Edge Detection", edge_scharr)
cv2.imshow("Roberts Edge Detection", edge_roberts)
cv2.waitKey(0)
cv2.destroyAllWindows()

gaussian_img2 = ndi.gaussian_filter(gray, sigma=3)
gaussian_img21 = np.reshape(gaussian_img2, -1)
df['Gaussian_blur3'] = gaussian_img21

median_img1 = ndi.median_filter(gray, size=3)
median_img12 = np.reshape(median_img1, -1)
df['Median_filter3'] = median_img12

variance_img = ndi.generic_filter(gray, np.var, size=3)
variance_img2 = np.reshape(variance_img, -1)
df['Variance_filter3'] = variance_img2

local_contrast = gray - gaussian_img2
local_contrast2 = np.reshape(local_contrast, -1)
df['Local_contrast'] = local_contrast2

local_median = ndi.median_filter(local_contrast, size=3)
local_median2 = np.reshape(local_median, -1)
df['Local_median'] = local_median2

cv2.imshow("Local Contrast", local_contrast)
cv2.imshow("Local Median", local_median)
cv2.imshow("Gaussian Blur sigma=3", ndi.gaussian_filter(gray, sigma=3))
cv2.imshow("Median Filter size=3", ndi.median_filter(gray, size=3))
cv2.imshow("Variance Filter size=3", variance_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

mask_path = os.path.join(script_dir, '..', 'Datasets', 'DRIVE', 'training', '1st_manual', '21_manual1.gif')
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
labels = (mask > 0).astype(int).flatten()
df['Label'] = labels

print(df.head())
print(f"Labels array: {labels}")

##################################################################################################################

Y_train = df['Label'].values
X_train = df.drop(labels=['Label'], axis=1)

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, Y_train)

predictions = model.predict()

accuracy = metrics.accuracy_score(Y_train, predictions)
print(f"Training Accuracy: {accuracy}")
