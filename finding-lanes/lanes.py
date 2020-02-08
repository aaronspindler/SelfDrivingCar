import cv2
import numpy as np


def canny(image):
    # Step 1 Make image grayscale so there is only 1 channel instead of 3
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

    # Step 2 Reduce noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Step 3 Find the gradient
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image



image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)
cropped_image = region_of_interest(canny)

cv2.imshow('results', cropped_image)
cv2.waitKey(0)
