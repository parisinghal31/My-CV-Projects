import cv2
import numpy as np

# This function runs whenever you click the mouse
def inspect_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Get the BGR value (OpenCV reads images as BGR)
        # Note: Numpy uses (y, x) indexing for (row, column)
        bgr_pixel = img[y, x]
        b, g, r = bgr_pixel

        # 2. Convert pixel to other color spaces
        # We create a 1x1 image containing just this one pixel to use CV converters
        pixel_node = np.uint8([[bgr_pixel]])
        
        hsv = cv2.cvtColor(pixel_node, cv2.COLOR_BGR2HSV)[0][0]
        lab = cv2.cvtColor(pixel_node, cv2.COLOR_BGR2LAB)[0][0]
        gray = cv2.cvtColor(pixel_node, cv2.COLOR_BGR2GRAY)[0][0]

        # 3. Print the data
        print(f"\n--- Pixel at ({x}, {y}) ---")
        print(f"BGR:      {bgr_pixel}")
        print(f"HSV:      {hsv}")
        print(f"LAB:      {lab}")
        print(f"Grayscale: {gray}")

        # 4. Zoom Neighborhood (11x11)
        # We grab 5 pixels in every direction
        zoom = img[y-5:y+6, x-5:x+6]
        
        # Resize it so we can actually see it (scale up 20x)
        zoom_big = cv2.resize(zoom, (220, 220), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("11x11 Zoom", zoom_big)

# Load your image (put an image file named 'test.jpg' in your folder)
img = cv2.imread('test.png')

if img is None:
    print("Error: Could not find image!")
else:
    cv2.namedWindow('Pixel Inspector')
    cv2.setMouseCallback('Pixel Inspector', inspect_pixel)

    print("Click anywhere on the image. Press 'q' to exit.")
    
    while True:
        cv2.imshow('Pixel Inspector', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()