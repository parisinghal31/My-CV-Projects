# Classical Segmentation Suite

## Overview

This project compares four classical image segmentation
techniques:

1. K-Means Color Segmentation
2. Mean Shift Segmentation
3. GrabCut Segmentation
4. Watershed Segmentation

All methods are applied to the same cluttered scene and
their strengths and weaknesses are analyzed.

---

## Features

- K-Means clustering
- Mean Shift filtering
- GrabCut foreground extraction
- Watershed marker-based segmentation
- Side-by-side comparison visualization
- Failure mode analysis

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib

---

## Folder Structure

input/
outputs/
main.py
README.md

---

## Run

python main.py

---

## Output Files

kmeans.png
meanshift.png
grabcut.png
watershed.png
comparison.png
---

## Observations

### K-Means
Fast but merges similar colors.

### Mean Shift
Produces smoother regions but may remove details.

### GrabCut
Best foreground extraction but requires user initialization.

### Watershed
Strong boundary detection but often over-segments.

---

## Conclusion

No segmentation algorithm performs best in all situations.
K-Means is fastest, GrabCut gives the cleanest foreground,
Mean Shift produces smooth regions, and Watershed excels
at boundary detection.
