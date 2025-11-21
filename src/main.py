import numpy as np
import pandas as pd
import cv2
import os
from skimage.filters import sobel, prewitt, scharr, roberts
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
train_images_path = os.path.join(script_dir, '..', 'Datasets', 'DRIVE', 'training', 'images')
train_labels_path = os.path.join(script_dir, '..', 'Datasets', 'DRIVE', 'training', '1st_manual')
output_path_train = os.path.join(script_dir, 'output_train.csv')

test_images_path = os.path.join(script_dir, '..', 'Datasets', 'DRIVE', 'test', 'images')
output_path_test = os.path.join(script_dir, 'output_test.csv')

image_files = sorted(glob.glob(os.path.join(train_images_path, '*.tif')))
print(f"Found {len(image_files)} training images")

all_dfs = []
for image_file in image_files:

    fname = os.path.basename(image_file)
    print(f"Processing {fname}...")

    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Could not read the image.")
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    """
    cv2.imshow("Image", image)
    cv2.imshow("Gray Image", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
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
                    #print(  f'Gabor_label: {gabor_label}, theta = {theta}, sigma = {sigma}, lambda = {lambd}, gamma = {gamma}' )
                    num += 1

    # Canny edge detector
    edge = cv2.Canny(gray, 35, 200)
    """
    cv2.imshow("Gray Image for Canny", gray)
    cv2.imshow("Canny Edge Detection", edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
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

    """
    cv2.imshow("Sobel Edge Detection", edge_sobel)
    cv2.imshow("Prewitt Edge Detection", edge_prewitt)
    cv2.imshow("Scharr Edge Detection", edge_scharr)
    cv2.imshow("Roberts Edge Detection", edge_roberts)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    gaussian_img2 = ndi.gaussian_filter(gray, sigma=3)
    gaussian_img21 = np.reshape(gaussian_img2, -1)
    df['Gaussian_blur3'] = gaussian_img21

    median_img1 = ndi.median_filter(gray, size=3)
    #cv2.imshow("Local median", median_img1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    median_img12 = np.reshape(median_img1, -1)
    df['Median_filter3'] = median_img12

    variance_img = ndi.generic_filter(gray, np.var, size=3)
    #cv2.imshow("Variance filter", variance_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    variance_img2 = np.reshape(variance_img, -1)
    df['Variance_filter3'] = variance_img2

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    local_contrast = gray - gaussian_img2

    #closed = cv2.morphologyEx(local_contrast, cv2.MORPH_CLOSE, None, iterations=1)
    opened = cv2.morphologyEx(local_contrast, cv2.MORPH_OPEN, kernel, iterations=1)

    local_contrast = opened
    local_contrast2 = local_contrast.reshape(-1)
    #cv2.imshow("Local contrast", local_contrast)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    df['Local_contrast'] = local_contrast2

    local_median = ndi.median_filter(local_contrast, size=3)
    #cv2.imshow("Local median", local_median)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    local_median2 = np.reshape(local_median, -1)
    df['Local_median'] = local_median2
    
    """
    cv2.imshow("Local Contrast", local_contrast)
    cv2.imshow("Local Median", local_median)
    cv2.imshow("Gaussian Blur sigma=3", ndi.gaussian_filter(gray, sigma=3))
    cv2.imshow("Median Filter size=3", ndi.median_filter(gray, size=3))
    cv2.imshow("Variance Filter size=3", variance_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    mask_name = fname.replace('_training.tif', '_manual1.gif')
    mask_path = os.path.join(train_labels_path, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Warning: No mask found for {fname}")
        continue

    df['Label'] = (mask > 0).astype(int).flatten()
    all_dfs.append(df)
    print(f"Done: {fname}")

df_train = pd.concat(all_dfs, ignore_index=True)
print(f"Total pixels in training data: {len(df_train)}")
print(f"Total features per pixel: {df_train.shape[1]-1}")

df_train.to_csv(output_path_train, index=False)
print("Training data saved")

##################################################################################################################

all_dfs_test = []
image_files_test = sorted(glob.glob(os.path.join(test_images_path, '*.tif')))
for image_file in image_files_test:

    fname = os.path.basename(image_file)
    print(f"Processing {fname}...")

    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Could not read the image.")
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    df = pd.DataFrame()
    image2 = np.reshape(gray, -1)
    df['pixel_value'] = image2

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
                    #print(  f'Gabor_label: {gabor_label}, theta = {theta}, sigma = {sigma}, lambda = {lambd}, gamma = {gamma}' )
                    num += 1

    edge = cv2.Canny(gray, 35, 200)
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

    gaussian_img2 = ndi.gaussian_filter(gray, sigma=3)
    gaussian_img21 = np.reshape(gaussian_img2, -1)
    df['Gaussian_blur3'] = gaussian_img21

    median_img1 = ndi.median_filter(gray, size=3)
    median_img12 = np.reshape(median_img1, -1)
    df['Median_filter3'] = median_img12

    variance_img = ndi.generic_filter(gray, np.var, size=3)
    variance_img2 = np.reshape(variance_img, -1)
    df['Variance_filter3'] = variance_img2

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    local_contrast = gray - gaussian_img2

    #closed = cv2.morphologyEx(local_contrast, cv2.MORPH_CLOSE, None, iterations=1)
    opened = cv2.morphologyEx(local_contrast, cv2.MORPH_OPEN, kernel, iterations=1)

    local_contrast = opened
    local_contrast2 = local_contrast.reshape(-1)
    df['Local_contrast'] = local_contrast2

    local_median = ndi.median_filter(local_contrast, size=3)
    local_median2 = np.reshape(local_median, -1)
    df['Local_median'] = local_median2

    all_dfs_test.append(df)
    print(f"Done: {fname}")

df_test = pd.concat(all_dfs_test, ignore_index=True)
print(f"Total pixels in training data: {len(df_test)}")
print(f"Total features per pixel: {df_test.shape[1]-1}")

df_test.to_csv(output_path_test, index=False)
print("Testing data saved")