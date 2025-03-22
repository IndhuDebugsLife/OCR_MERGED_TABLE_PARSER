import cv2
import numpy as np


# Load the image
image_path = 'image_0b42a5.png'  # Replace with the correct path if needed
img = cv2.imread(image_path)

# Rectangle coordinates
x1 = 108
y1 = 1269
width = 453
height = 140

# Calculate the bottom-right coordinates
x2 = x1 + width
y2 = y1 + height

# Draw the rectangle
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle, 2 pixels thick

# Display the image with the rectangle
cv2_imshow(img)