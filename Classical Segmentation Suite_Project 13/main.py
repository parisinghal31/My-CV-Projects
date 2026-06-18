import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "sample input image.jpg"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise Exception(
        f"Could not load image:\n{IMAGE_PATH}"
    )

image = cv2.resize(image, (1200, 800))

print("\nImage Loaded Successfully")
print("Image Shape:", image.shape)

# ============================================================
# K-MEANS SEGMENTATION
# ============================================================

def kmeans_segmentation(img, k=8):

    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    _, labels, centers = cv2.kmeans(
        pixel_values,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)

    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(img.shape)

    return segmented

# ============================================================
# MEAN SHIFT SEGMENTATION
# ============================================================

def mean_shift_segmentation(img):

    segmented = cv2.pyrMeanShiftFiltering(
        img,
        sp=25,
        sr=35
    )

    return segmented

# ============================================================
# GRABCUT SEGMENTATION
# ============================================================

def grabcut_segmentation(img):

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (
        80,
        80,
        img.shape[1] - 160,
        img.shape[0] - 160
    )

    cv2.grabCut(
        img,
        mask,
        rect,
        bgdModel,
        fgdModel,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    mask2 = np.where(
        (mask == cv2.GC_BGD) |
        (mask == cv2.GC_PR_BGD),
        0,
        1
    ).astype("uint8")

    segmented = img * mask2[:, :, np.newaxis]

    return segmented

# ============================================================
# WATERSHED SEGMENTATION
# ============================================================

def watershed_segmentation(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV +
        cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    sure_bg = cv2.dilate(
        opening,
        kernel,
        iterations=3
    )

    dist_transform = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    _, sure_fg = cv2.threshold(
        dist_transform,
        0.35 * dist_transform.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(
        sure_bg,
        sure_fg
    )

    _, markers = cv2.connectedComponents(
        sure_fg
    )

    markers = markers + 1

    markers[unknown == 255] = 0

    watershed_img = img.copy()

    markers = cv2.watershed(
        watershed_img,
        markers
    )

    watershed_img[markers == -1] = [0, 0, 255]

    return watershed_img

# ============================================================
# RUN SEGMENTATION
# ============================================================

print("\nRunning K-Means...")
start = time.time()
kmeans_result = kmeans_segmentation(image)
kmeans_time = time.time() - start

print("Running Mean Shift...")
start = time.time()
meanshift_result = mean_shift_segmentation(image)
meanshift_time = time.time() - start

print("Running GrabCut...")
start = time.time()
grabcut_result = grabcut_segmentation(image)
grabcut_time = time.time() - start

print("Running Watershed...")
start = time.time()
watershed_result = watershed_segmentation(image)
watershed_time = time.time() - start

# ============================================================
# SAVE OUTPUTS
# ============================================================

kmeans_path = os.path.join(
    OUTPUT_DIR,
    "kmeans.png"
)

meanshift_path = os.path.join(
    OUTPUT_DIR,
    "meanshift.png"
)

grabcut_path = os.path.join(
    OUTPUT_DIR,
    "grabcut.png"
)

watershed_path = os.path.join(
    OUTPUT_DIR,
    "watershed.png"
)

comparison_path = os.path.join(
    OUTPUT_DIR,
    "comparison.png"
)

cv2.imwrite(kmeans_path, kmeans_result)
cv2.imwrite(meanshift_path, meanshift_result)
cv2.imwrite(grabcut_path, grabcut_result)
cv2.imwrite(watershed_path, watershed_result)

print("\nSegmentation outputs saved successfully.")

# ============================================================
# EXECUTION TIME REPORT
# ============================================================

print("\n==============================")
print("EXECUTION TIME REPORT")
print("==============================")

print(f"K-Means    : {kmeans_time:.3f} sec")
print(f"Mean Shift : {meanshift_time:.3f} sec")
print(f"GrabCut    : {grabcut_time:.3f} sec")
print(f"Watershed  : {watershed_time:.3f} sec")

# ============================================================
# COMPARISON VISUALIZATION
# ============================================================

plt.figure(figsize=(18, 10))

titles = [
    "Original",
    "K-Means",
    "Mean Shift",
    "GrabCut",
    "Watershed"
]

images = [
    image,
    kmeans_result,
    meanshift_result,
    grabcut_result,
    watershed_result
]

for i in range(len(images)):

    plt.subplot(2, 3, i + 1)

    plt.imshow(
        cv2.cvtColor(
            images[i],
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    comparison_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# FINAL REPORT
# ============================================================

print("\n==============================")
print("OUTPUT FILES")
print("==============================")

print(kmeans_path)
print(meanshift_path)
print(grabcut_path)
print(watershed_path)
print(comparison_path)

print("\n==============================")
print("OBSERVATIONS")
print("==============================")

print("K-Means   : Color-based clustering.")
print("MeanShift : Better region preservation.")
print("GrabCut   : Foreground extraction method.")
print("Watershed : Strong boundary detection.")

print("\nProject Completed Successfully!")