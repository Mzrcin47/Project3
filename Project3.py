import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

image_path = 'data/motherboard_image.JPEG'
original_image = cv2.imread(image_path)
original_image = cv2.resize(original_image, (800, 600))

# Convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Thresholding
_, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Dilation
dilated_image = cv2.dilate(thresholded_image, None, iterations=2)

# Contour detection
contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area
min_contour_area = 500
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Create a mask
mask = np.zeros_like(gray_image, dtype=np.uint8)
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)
cv2.fillPoly(mask, filtered_contours, 255)

# Morphological closing
kernel = np.ones((15, 15), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Color masking
lower_color = np.array([0, 0, 150])  # Adjusted based on your color
upper_color = np.array([179, 20, 255])  # Adjusted based on your color
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
color_mask = cv2.inRange(hsv_image, lower_color, upper_color)
combined_mask = cv2.bitwise_and(color_mask, mask)

# Bitwise AND operation to extract the region of interest
result = cv2.bitwise_and(original_image, original_image, mask=combined_mask)

# Display images
cv2.imshow('Original Image', cv2.resize(original_image, (800, 600)))
cv2.imshow('Combined Mask', cv2.resize(combined_mask, (800, 600)))
cv2.imshow('Extracted Motherboard', cv2.resize(result, (800, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()