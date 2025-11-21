import pandas as pd
import os
import cv2
import numpy as np
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', 'predictions_test.csv')

df_pred = pd.read_csv(csv_path)
Y_pred = df_pred['Predicted_Label'].values

test_images_path = os.path.join(script_dir, '..', 'Datasets', 'DRIVE', 'test', 'images')
image_path = glob.glob(os.path.join(test_images_path, '*.tif'))

start_idx = 0 

for idx, test_image in enumerate(image_path):
    img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    num_pixels = h * w

    pred_slice = Y_pred[start_idx:start_idx + num_pixels]
    start_idx += num_pixels
    pred_mask = pred_slice.reshape(h, w)
    pred_mask_img = (pred_mask * 255).astype(np.uint8)

    cv2.imshow("Original Image", img)
    cv2.imshow("Predicted Mask", pred_mask_img)
    cv2.waitKey(0)

    output_path = f"pred_mask_{idx+1}.png"
    cv2.imwrite(output_path, pred_mask_img)
    print(f"Saved {output_path}")

cv2.destroyAllWindows()