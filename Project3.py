import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

image_path = 'data/motherboard_image.JPEG'
original_image = cv2.imread(image_path)
original_image = cv2.resize(original_image, (800, 600))

gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)


harris_response = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)
harris_response = cv2.dilate(harris_response, None)

harris_corners = np.zeros_like(harris_response)
harris_corners[harris_response > 0.01 * harris_response.max()] = 255


harris_corners = np.uint8(harris_corners)


dilated_corners = cv2.dilate(harris_corners, None, iterations=7)

contours, _ = cv2.findContours(dilated_corners, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_contour_area = 900
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

mask = np.zeros_like(gray_image, dtype=np.uint8)
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

kernel = np.ones((15, 15), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

lower_color = np.array([0, 0, 50])
upper_color = np.array([200, 200, 255])
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
color_mask = cv2.inRange(hsv_image, lower_color, upper_color)

combined_mask = cv2.bitwise_and(color_mask, mask)

result = cv2.bitwise_and(original_image, original_image, mask=combined_mask)

cv2.imshow('Original Image', cv2.resize(original_image, (800, 600)))
cv2.imshow('Harris Corners', cv2.resize(dilated_corners, (800, 600)))
cv2.imshow('Combined Mask', cv2.resize(combined_mask, (800, 600)))
cv2.imshow('Extracted Motherboard', cv2.resize(result, (800, 600)))

cv2.waitKey(0)
cv2.destroyAllWindows()