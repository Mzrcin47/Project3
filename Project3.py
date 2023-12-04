import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

image_path = 'data/motherboard_image.JPEG'
original_image = cv2.imread(image_path)


original_image = cv2.resize(original_image, (800, 600))


gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

_, thresholded_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(thresholded_image, 30, 100)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_contour_area = 500
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

mask = np.zeros_like(gray_image, dtype=np.uint8)

if filtered_contours:

    largest_contour = max(filtered_contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)


original_image = original_image.astype(np.uint8)
mask = mask.astype(np.uint8)

result = cv2.bitwise_and(original_image, original_image, mask=mask)


cv2.imshow('Original Image', cv2.resize(original_image, (800, 600)))
cv2.imshow('Mask', cv2.resize(mask, (800, 600)))
cv2.imshow('Extracted Motherboard', cv2.resize(result, (800, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()