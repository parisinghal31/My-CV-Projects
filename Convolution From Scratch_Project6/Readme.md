# 2D Image Convolution Engine From Scratch:

-A pure Python and NumPy engineering implementation of 2D image convolutions from scratch.
-This project implements classic computer vision filters without using high-level deep learning abstraction libraries, demonstrating exactly how Convolutional Neural Network (CNN) layers handle spatial matrices.

# Core Features:

- Multi-Channel Processing Layers: Dynamically breaks down and handles both grayscale matrices and 3D color images (BGR/RGB).
- CNN Spatial Sizing Formula: Uses explicit padding and stride mechanisms to compute strict output resolutions.
- Mathematical Integrity: Implements floating-point (`float32`) tracking to prevent integer overflow and underflow wrap-around artifacts.

# Handcrafted Filter Kernels:

- The project builds and evaluates five industry-standard matrix weights by hand:
1. Box Blur: Uniform local neighborhood averaging.
2. Gaussian Blur: Weighted distribution smoothing for clean noise reduction.
3. Sharpen: Spatial center amplification to accent high-frequency structures.
4. Emboss: Asymmetrical diagonal edge manipulation producing a 3D shadow layout.
5. Sobel (Horizontal Gradient): Calculates the first derivative of image intensity to isolate clean vertical edges.

# Tools Used:

- Python
- NumPy
- OpenCV