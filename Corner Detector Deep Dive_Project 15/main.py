import cv2
import numpy as np
import os

# =====================================================
# CONFIGURATION
# =====================================================

IMAGE_PATH = r"C:\Users\Khushboo\Desktop\CV Projects\Corner Detector Deep Dive_Project 15\chessboardtessellation_input.jpg"

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD IMAGE
# =====================================================

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise Exception(
        f"\nCould not load image:\n{IMAGE_PATH}\n"
    )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("\n[INFO] Image Loaded Successfully")
print("[INFO] Shape:", gray.shape)

# =====================================================
# STEP 1 : COMPUTE IMAGE GRADIENTS
# =====================================================

Ix = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

Iy = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

# =====================================================
# STEP 2 : STRUCTURE TENSOR
# =====================================================

Ixx = Ix * Ix
Iyy = Iy * Iy
Ixy = Ix * Iy

Ixx = cv2.GaussianBlur(Ixx, (5, 5), 1)
Iyy = cv2.GaussianBlur(Iyy, (5, 5), 1)
Ixy = cv2.GaussianBlur(Ixy, (5, 5), 1)

# =====================================================
# STEP 3 : EIGENVALUES
# =====================================================

trace = Ixx + Iyy

det = (Ixx * Iyy) - (Ixy * Ixy)

temp = np.sqrt(
    np.maximum(
        trace * trace - 4 * det,
        0
    )
)

lambda1 = (trace + temp) / 2
lambda2 = (trace - temp) / 2

# =====================================================
# SAVE HEATMAP FUNCTION
# =====================================================

def save_heatmap(data, path):

    normalized = cv2.normalize(
        data,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    normalized = normalized.astype(np.uint8)

    heatmap = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )

    cv2.imwrite(path, heatmap)

    return heatmap

# =====================================================
# λ1 HEATMAP
# =====================================================

heat_lambda1 = save_heatmap(
    lambda1,
    os.path.join(
        OUTPUT_DIR,
        "lambda1_heatmap.jpg"
    )
)

# =====================================================
# λ2 HEATMAP
# =====================================================

heat_lambda2 = save_heatmap(
    lambda2,
    os.path.join(
        OUTPUT_DIR,
        "lambda2_heatmap.jpg"
    )
)

print("[INFO] Eigenvalue Heatmaps Saved")

# =====================================================
# STEP 4 : HARRIS RESPONSE
# =====================================================

k = 0.04

R = det - k * (trace ** 2)

heat_harris = save_heatmap(
    R,
    os.path.join(
        OUTPUT_DIR,
        "harris_response_heatmap.jpg"
    )
)

# =====================================================
# STEP 5 : HARRIS CORNERS + NMS
# =====================================================

harris_img = img.copy()

threshold_harris = 0.02 * R.max()

harris_count = 0

for y in range(1, R.shape[0] - 1):

    for x in range(1, R.shape[1] - 1):

        if R[y, x] > threshold_harris:

            patch = R[
                y - 1:y + 2,
                x - 1:x + 2
            ]

            if R[y, x] == patch.max():

                cv2.circle(
                    harris_img,
                    (x, y),
                    4,
                    (0, 0, 255),
                    -1
                )

                harris_count += 1

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "harris.jpg"
    ),
    harris_img
)

print(
    "[INFO] Harris Corners:",
    harris_count
)

# =====================================================
# STEP 6 : SHI TOMASI
# =====================================================

shi_response = np.minimum(
    lambda1,
    lambda2
)

shi_img = img.copy()

threshold_shi = 0.05 * shi_response.max()

shi_count = 0

for y in range(
        1,
        shi_response.shape[0] - 1
):

    for x in range(
            1,
            shi_response.shape[1] - 1
    ):

        if shi_response[y, x] > threshold_shi:

            patch = shi_response[
                y - 1:y + 2,
                x - 1:x + 2
            ]

            if shi_response[y, x] == patch.max():

                cv2.circle(
                    shi_img,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

                shi_count += 1

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "shi_tomasi.jpg"
    ),
    shi_img
)

print(
    "[INFO] Shi-Tomasi Corners:",
    shi_count
)

# =====================================================
# STEP 7 : FAST DETECTOR
# =====================================================

fast = cv2.FastFeatureDetector_create(
    threshold=20,
    nonmaxSuppression=True
)

keypoints = fast.detect(
    gray,
    None
)

fast_img = cv2.drawKeypoints(
    img,
    keypoints,
    None,
    color=(255, 0, 0)
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "fast.jpg"
    ),
    fast_img
)

print(
    "[INFO] FAST Keypoints:",
    len(keypoints)
)

# =====================================================
# STEP 8 : COMPARISON PANEL
# =====================================================

harris_resized = cv2.resize(
    harris_img,
    (600, 450)
)

shi_resized = cv2.resize(
    shi_img,
    (600, 450)
)

fast_resized = cv2.resize(
    fast_img,
    (600, 450)
)

lambda_resized = cv2.resize(
    heat_lambda1,
    (600, 450)
)

top = np.hstack(
    [
        harris_resized,
        shi_resized
    ]
)

bottom = np.hstack(
    [
        fast_resized,
        lambda_resized
    ]
)

comparison = np.vstack(
    [
        top,
        bottom
    ]
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "comparison.jpg"
    ),
    comparison
)

# =====================================================
# DISPLAY RESULTS
# =====================================================

cv2.imshow(
    "Corner Detector Comparison",
    comparison
)

cv2.waitKey(0)
cv2.destroyAllWindows()

# =====================================================
# SUMMARY
# =====================================================

print("\n")
print("=" * 50)
print("PROJECT 15 RESULTS")
print("=" * 50)

print("Harris Corners   :", harris_count)
print("Shi-Tomasi       :", shi_count)
print("FAST Keypoints   :", len(keypoints))

print("=" * 50)

print("\nOutput Folder:")

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)