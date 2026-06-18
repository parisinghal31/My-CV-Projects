# Corner Detector Deep Dive – Harris, Shi-Tomasi & FAST

## Project Overview

This project explores three important corner detection algorithms in Computer Vision:

1. Harris Corner Detector (implemented from scratch)
2. Shi-Tomasi Corner Detector (implemented from scratch)
3. FAST Corner Detector (using OpenCV)

The project demonstrates how image gradients, structure tensors, eigenvalues, and corner response functions can be used to identify highly distinctive points in an image.

A chessboard image is used as the test image because chessboard intersections are ideal examples of strong corners.

---

# Objectives

The primary objectives of this project are:

* Understand the concept of image corners.
* Compute image gradients using Sobel operators.
* Construct the Structure Tensor.
* Compute eigenvalues of the Structure Tensor.
* Implement Harris Corner Detector from scratch.
* Implement Shi-Tomasi Corner Detector from scratch.
* Apply Non-Maximum Suppression (NMS).
* Compare Harris and Shi-Tomasi with FAST.
* Visualize Structure Tensor eigenvalues using heatmaps.
* Analyze corner detector performance.

---

# What is a Corner?

A corner is a point in an image where intensity changes significantly in multiple directions.

Unlike flat regions or edges, corners provide strong and distinctive features that are useful for:

* Object Recognition
* Feature Matching
* Image Stitching
* Camera Calibration
* Visual SLAM
* Motion Tracking
* Robotics Navigation

Examples:

### Flat Region

Little change in any direction.

```
--------
--------
--------
```

### Edge

Large change in one direction only.

```
████████
████████
--------
--------
```

### Corner

Large change in all directions.

```
████----
████----
--------
--------
```

---

# Theory

## Step 1: Image Gradients

The first step is computing intensity changes along horizontal and vertical directions.

Horizontal Gradient:

Ix = dI/dx

Vertical Gradient:

Iy = dI/dy

These gradients are computed using Sobel operators.

```python
Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
```

---

## Step 2: Structure Tensor

The Structure Tensor captures local intensity variations.

For each pixel:

[
M =
\begin{bmatrix}
I_x^2 & I_xI_y \
I_xI_y & I_y^2
\end{bmatrix}
]

To reduce noise, Gaussian smoothing is applied:

[
M =
\begin{bmatrix}
S(I_x^2) & S(I_xI_y) \
S(I_xI_y) & S(I_y^2)
\end{bmatrix}
]

where S represents Gaussian filtering.

---

## Step 3: Eigenvalues of Structure Tensor

The eigenvalues λ₁ and λ₂ describe intensity variation in two principal directions.

### Flat Region

λ₁ ≈ 0

λ₂ ≈ 0

### Edge

λ₁ is large

λ₂ is small

### Corner

λ₁ is large

λ₂ is large

Therefore, corners are locations where both eigenvalues have high values.

---

# Harris Corner Detector

Harris Corner Detector uses the response:

[
R = det(M) - k(trace(M)^2)
]

where:

[
det(M) = \lambda_1\lambda_2
]

[
trace(M) = \lambda_1 + \lambda_2
]

and

k = 0.04

Interpretation:

### Flat Region

R ≈ 0

### Edge

R < 0

### Corner

R >> 0

Large positive values correspond to corners.

---

# Non-Maximum Suppression (NMS)

The Harris response often produces clusters of nearby corner pixels.

Non-Maximum Suppression retains only the strongest response within a neighborhood.

Benefits:

* Removes duplicate corner detections.
* Produces cleaner results.
* Improves visualization.

---

# Shi-Tomasi Corner Detector

Shi-Tomasi improves Harris by directly using eigenvalues.

Response:

[
R = min(\lambda_1,\lambda_2)
]

A corner is accepted only if both eigenvalues are large.

Advantages:

* More stable than Harris.
* Better feature selection.
* Widely used in KLT tracking.

---

# FAST Corner Detector

FAST stands for:

Features from Accelerated Segment Test

Instead of gradients and eigenvalues, FAST examines a circle of 16 pixels around a candidate pixel.

A point is classified as a corner if enough neighboring pixels are significantly brighter or darker than the center pixel.

Advantages:

* Extremely fast.
* Suitable for real-time applications.
* Used in ORB feature detector.

Disadvantages:

* Less mathematically rigorous.
* More sensitive to threshold selection.

---

# Algorithm Workflow

1. Read input image.
2. Convert image to grayscale.
3. Compute Sobel gradients.
4. Construct Structure Tensor.
5. Apply Gaussian smoothing.
6. Compute eigenvalues.
7. Generate eigenvalue heatmaps.
8. Compute Harris response.
9. Apply NMS.
10. Detect Harris corners.
11. Compute Shi-Tomasi response.
12. Apply NMS.
13. Detect Shi-Tomasi corners.
14. Detect FAST keypoints.
15. Save results and comparison image.

---

# Input Image

The project uses:

```
chessboardtessellation_input.jpg
```

Why chessboard?

* Contains many strong corners.
* Provides clean geometric structure.
* Frequently used in camera calibration.
* Ideal for evaluating corner detectors.

---

# Output Files

The program generates:

```
outputs/
│
├── harris.jpg
├── shi_tomasi.jpg
├── fast.jpg
├── lambda1_heatmap.jpg
├── lambda2_heatmap.jpg
├── harris_response_heatmap.jpg
└── comparison.jpg
```

---

# Output Description

## harris.jpg

Displays Harris corners using red markers.

---

## shi_tomasi.jpg

Displays Shi-Tomasi corners using green markers.

---

## fast.jpg

Displays FAST keypoints using blue markers.

---

## lambda1_heatmap.jpg

Heatmap of the first eigenvalue.

Bright regions indicate strong directional variation.

---

## lambda2_heatmap.jpg

Heatmap of the second eigenvalue.

Bright regions indicate strong variation in another principal direction.

---

## harris_response_heatmap.jpg

Visualization of Harris response values.

Bright areas correspond to potential corners.

---

## comparison.jpg

Combined visualization showing:

* Harris Detector
* Shi-Tomasi Detector
* FAST Detector
* Eigenvalue Heatmap

---

# Expected Results

For the chessboard image:

### Harris Detector

Detects strong chessboard intersections.

### Shi-Tomasi Detector

Produces cleaner and more stable corners.

### FAST Detector

Detects many keypoints quickly.

### Eigenvalue Heatmaps

Bright spots appear exactly at chessboard intersections.

---

# Applications

Corner detection is used in:

* Image Stitching
* Panorama Creation
* Camera Calibration
* Robotics
* Augmented Reality
* Visual SLAM
* Object Tracking
* Autonomous Vehicles
* Feature Matching

---

# Advantages and Limitations

## Harris

Advantages:

* Accurate
* Robust
* Theoretically strong

Limitations:

* Computationally expensive
* Not scale invariant

---

## Shi-Tomasi

Advantages:

* Better feature selection
* Stable tracking points

Limitations:

* Slightly slower than FAST

---

## FAST

Advantages:

* Very fast
* Real-time capable

Limitations:

* Sensitive to threshold
* Less descriptive

---

# Conclusion

This project successfully demonstrates the complete pipeline of classical corner detection techniques. Starting from image gradients and Structure Tensor analysis, Harris and Shi-Tomasi detectors are implemented from scratch and compared against FAST.

The eigenvalue heatmaps clearly show that corners occur at locations where both eigenvalues are high. Harris uses a response function based on determinant and trace, while Shi-Tomasi directly uses the minimum eigenvalue. FAST achieves much higher speed by avoiding gradient and matrix computations altogether.

The chessboard image provides a clear demonstration of how these detectors identify strong corner features, making this project an excellent introduction to feature detection in Computer Vision.
