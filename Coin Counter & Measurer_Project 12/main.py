import cv2
import numpy as np
import math

# =====================================================
# CONFIGURATION
# =====================================================

IMAGE_PATH = r"C:\Users\Khushboo\Desktop\CV Projects\Coin Counter & Measurer_Project 12\coins.png"

CARD_WIDTH_MM = 85.60

COIN_DATABASE = {
    20.0: 1,
    23.0: 2,
    25.0: 5,
    27.0: 10
}

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def classify_coin(diameter_mm):
    nearest = min(
        COIN_DATABASE.keys(),
        key=lambda x: abs(x - diameter_mm)
    )
    return COIN_DATABASE[nearest]

def circularity(contour):

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter == 0:
        return 0

    return (
        4 * np.pi * area
    ) / (perimeter * perimeter)

# =====================================================
# LOAD IMAGE
# =====================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Cannot load image: {IMAGE_PATH}"
    )

output = image.copy()

# =====================================================
# PREPROCESSING
# =====================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    21,
    5
)

kernel = np.ones(
    (3, 3),
    np.uint8
)

thresh = cv2.morphologyEx(
    thresh,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

# =====================================================
# FIND CONTOURS
# =====================================================

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# =====================================================
# DETECT REFERENCE CARD
# =====================================================

reference = None
max_area = 0

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 10000:
        continue

    peri = cv2.arcLength(
        cnt,
        True
    )

    approx = cv2.approxPolyDP(
        cnt,
        0.02 * peri,
        True
    )

    if len(approx) == 4:

        if area > max_area:

            max_area = area
            reference = approx

if reference is None:

    raise Exception(
        "Reference card not found."
    )

cv2.drawContours(
    output,
    [reference],
    -1,
    (255, 0, 0),
    3
)

rect = cv2.minAreaRect(
    reference
)

width_pixels = max(
    rect[1][0],
    rect[1][1]
)

pixels_per_mm = (
    width_pixels /
    CARD_WIDTH_MM
)

print(
    f"Pixels/mm: {pixels_per_mm:.3f}"
)

# =====================================================
# COIN DETECTION
# =====================================================

total_value = 0
coin_count = 0

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 1000:
        continue

    circ = circularity(cnt)

    if circ < 0.75:
        continue

    (x, y), radius = cv2.minEnclosingCircle(
        cnt
    )

    diameter_pixels = radius * 2

    diameter_mm = (
        diameter_pixels /
        pixels_per_mm
    )

    value = classify_coin(
        diameter_mm
    )

    total_value += value
    coin_count += 1

    center = (
        int(x),
        int(y)
    )

    radius = int(radius)

    cv2.circle(
        output,
        center,
        radius,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"Rs{value}",
        (
            center[0] - 20,
            center[1] - 10
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    cv2.putText(
        output,
        f"{diameter_mm:.1f} mm",
        (
            center[0] - 40,
            center[1] + 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2
    )

# =====================================================
# DISPLAY SUMMARY
# =====================================================

cv2.putText(
    output,
    f"Coins: {coin_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 255),
    2
)

cv2.putText(
    output,
    f"Total: Rs {total_value}",
    (20, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 255),
    2
)

print("\n========== RESULT ==========")
print(f"Coins Detected : {coin_count}")
print(f"Total Value    : Rs {total_value}")
print("============================")

cv2.imwrite(
    "output/result.png",
    output
)

cv2.imshow(
    "Coin Counter & Measurer",
    output
)

cv2.waitKey(0)
cv2.destroyAllWindows()