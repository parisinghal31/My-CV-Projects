import cv2
import numpy as np

def nothing(x):
    pass

# 1. Create a blank image or load one
# If you have an image, use: img = cv2.imread('your_image.jpg')
# Otherwise, let's create a colorful test pattern:
img = np.zeros((300, 512, 3), np.uint8)
for i in range(512):
    img[:, i] = [i//2, 128, 255 - i//2]

cv2.namedWindow('Color Space Explorer')

# 2. Create Trackbars for HSV Color Picking
cv2.createTrackbar('H_Low', 'Color Space Explorer', 0, 179, nothing)
cv2.createTrackbar('S_Low', 'Color Space Explorer', 0, 255, nothing)
cv2.createTrackbar('V_Low', 'Color Space Explorer', 0, 255, nothing)
cv2.createTrackbar('H_High', 'Color Space Explorer', 179, 179, nothing)
cv2.createTrackbar('S_High', 'Color Space Explorer', 255, 255, nothing)
cv2.createTrackbar('V_High', 'Color Space Explorer', 255, 255, nothing)

while True:
    # Get current trackbar positions
    h_l = cv2.getTrackbarPos('H_Low', 'Color Space Explorer')
    s_l = cv2.getTrackbarPos('S_Low', 'Color Space Explorer')
    v_l = cv2.getTrackbarPos('V_Low', 'Color Space Explorer')
    h_h = cv2.getTrackbarPos('H_High', 'Color Space Explorer')
    s_h = cv2.getTrackbarPos('S_High', 'Color Space Explorer')
    v_h = cv2.getTrackbarPos('V_High', 'Color Space Explorer')

    # Convert image to various spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Color Masking (The Segmenter)
    lower_range = np.array([h_l, s_l, v_l])
    upper_range = np.array([h_h, s_h, v_h])
    mask = cv2.inRange(hsv, lower_range, upper_range)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Show images
    cv2.imshow('Original', img)
    cv2.imshow('HSV Space', hsv)
    cv2.imshow('Mask (Black/White)', mask)
    cv2.imshow('Isolated Color', result)

    if cv2.waitKey(1) & 0xFF == 27: # Press 'Esc' to exit
        break

cv2.destroyAllWindows()