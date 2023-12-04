import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

image_path = 'data/motherboard_image.JPEG'
original_image = cv2.imread(image_path)

original_image = cv2.resize(original_image, (800, 600))

gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

_, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

dilated_image = cv2.dilate(thresholded_image, None, iterations=2)

# Find contours
contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_contour_area = 5000
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

mask = np.zeros_like(gray_image, dtype=np.uint8)
cv2.fillPoly(mask, filtered_contours, 255)

kernel = np.ones((25, 25), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

original_image = original_image.astype(np.uint8)
mask = mask.astype(np.uint8)

result = cv2.bitwise_and(original_image, original_image, mask=mask)

cv2.imshow('Original Image', cv2.resize(original_image, (800, 600)))
cv2.imshow('Mask', cv2.resize(mask, (800, 600)))
cv2.imshow('Extracted Motherboard', cv2.resize(result, (800, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()