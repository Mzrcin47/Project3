import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# Step 1 - Object Masking

image_path = 'data/motherboard_image.JPEG'
original_image = cv2.imread(image_path)
original_image = np.uint8(original_image)

_, thresholded_image = cv2.threshold(original_image, 150, 255, cv2.THRESH_BINARY)
