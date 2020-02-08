import cv2
import numpy as np

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

# Step 1 Make image grayscale so there is only 1 channel instead of 3
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

# Step 2 Reduce noise
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Step 3 Find the gradient
canny = cv2.Canny(blur, 50, 150)




cv2.imshow('results', canny)
cv2.waitKey(0)
