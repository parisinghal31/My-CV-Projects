import cv2
import numpy as np

# 1. Create two test images (gradients)
# Image A: Horizontal gradient (0 to 255)
# Image B: Vertical gradient (0 to 255)
image_a = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
image_b = image_a.T  # Transpose to make it vertical

# --- ARITHMETIC FROM SCRATCH (THE NUMPY WAY) ---

# Standard addition (Wraparound/Modulo)
# 200 + 100 becomes 44
numpy_add = image_a + image_b

# Standard subtraction
# 50 - 100 becomes 206 (wraps around the bottom)
numpy_sub = image_a - image_b

# --- THE CORRECT CV WAY (SATURATION) ---

# OpenCV addition (Saturates at 255)
# 200 + 100 becomes 255
cv_add = cv2.add(image_a, image_b)

# Manual Saturation Addition using NumPy
# We convert to a larger container (int16) to do the math, then clip
scratch_add = image_a.astype(np.int16) + image_b.astype(np.int16)
scratch_add = np.clip(scratch_add, 0, 255).astype(np.uint8)

# --- BLENDING (Linear Interpolation) ---
# Formula: (ImageA * alpha) + (ImageB * (1 - alpha))
alpha = 0.5
blended = cv2.addWeighted(image_a, alpha, image_b, 1 - alpha, 0)

# --- DISPLAY RESULTS ---
cv2.imshow("Original A", image_a)
cv2.imshow("Numpy Add (Modulo/Glitch)", numpy_add)
cv2.imshow("OpenCV Add (Saturated/Clean)", cv_add)
cv2.imshow("Scratch Saturation Add", scratch_add)
cv2.imshow("Blended", blended)

print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()