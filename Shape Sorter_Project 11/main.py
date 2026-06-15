import cv2
import numpy as np
from collections import defaultdict

# =====================================================
# CONFIGURATION
# =====================================================

IMAGE_PATH = r"C:\Users\Khushboo\Desktop\CV Projects\Shape Sorter_Project 11\SampleInputImage.jpg"

MIN_CONTOUR_AREA = 5000

# =====================================================
# HU MOMENTS
# =====================================================

def get_hu_moments(contour):
    moments = cv2.moments(contour)

    hu = cv2.HuMoments(moments)

    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return hu.flatten()


# =====================================================
# SHAPE CLASSIFICATION
# =====================================================

def classify_shape(contour):

    perimeter = cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    vertices = len(approx)

    area = cv2.contourArea(contour)

    # ------------------------------------
    # Triangle
    # ------------------------------------

    if vertices == 3:
        return "Triangle", vertices

    # ------------------------------------
    # Square / Rectangle
    # ------------------------------------

    elif vertices == 4:

        x, y, w, h = cv2.boundingRect(approx)

        aspect_ratio = w / float(h)

        if 0.95 <= aspect_ratio <= 1.05:
            return "Square", vertices

        return "Rectangle", vertices

    # ------------------------------------
    # Pentagon
    # ------------------------------------

    elif vertices == 5:
        return "Pentagon", vertices

    # ------------------------------------
    # Hexagon
    # ------------------------------------

    elif vertices == 6:
        return "Hexagon", vertices

    # ------------------------------------
    # Circle / Oval
    # ------------------------------------

    circularity = (
        4 * np.pi * area /
        (perimeter * perimeter + 1e-10)
    )

    if len(contour) >= 5:

        ellipse = cv2.fitEllipse(contour)

        (_, _), (major_axis, minor_axis), _ = ellipse

        axis_ratio = (
            min(major_axis, minor_axis)
            /
            max(major_axis, minor_axis)
        )

        # Circle

        if circularity > 0.85 and axis_ratio > 0.92:
            return "Circle", vertices

        # Oval

        return "Oval", vertices

    return "Unknown", vertices


# =====================================================
# MAIN
# =====================================================

def main():

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Could not load image.")
        return

    output = image.copy()

    # =====================================================
    # HSV SEGMENTATION
    # =====================================================

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Keep colorful objects, remove gray background

    mask = cv2.inRange(
        hsv,
        (0, 40, 40),
        (180, 255, 255)
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # =====================================================
    # FIND CONTOURS
    # =====================================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    counts = defaultdict(int)

    print("\n" + "=" * 60)
    print("DETECTED SHAPES")
    print("=" * 60)

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_CONTOUR_AREA:
            continue

        shape, vertices = classify_shape(contour)

        counts[shape] += 1

        hu = get_hu_moments(contour)

        x, y, w, h = cv2.boundingRect(contour)

        # Draw contour

        cv2.drawContours(
            output,
            [contour],
            -1,
            (0, 255, 0),
            3
        )

        # Draw bounding box

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Label

        cv2.putText(
            output,
            shape,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        print(f"\nShape : {shape}")
        print(f"Vertices : {vertices}")
        print(f"Area : {area:.2f}")

        print("Hu Moments:")
        print(np.round(hu, 4))

    # =====================================================
    # SUMMARY
    # =====================================================

    print("\n" + "=" * 60)
    print("SHAPE COUNTS")
    print("=" * 60)

    total = 0

    for shape, count in sorted(counts.items()):
        print(f"{shape:<10} : {count}")
        total += count

    print(f"\nTotal Shapes : {total}")

    # =====================================================
    # SAVE OUTPUT
    # =====================================================

    output_path = "Detected_Shapes_Output.jpg"

    cv2.imwrite(
        output_path,
        output
    )

    print("\nOutput image saved as:")
    print(output_path)

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    cv2.imshow("Original Image", image)
    cv2.imshow("HSV Mask", mask)
    cv2.imshow("Detected Shapes", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()