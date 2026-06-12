# Edge Detector Implementation from First Principles

- An educational computer vision suite executing fundamental edge detection algorithms alongside a complete, modular implementation of the **Canny Edge Detector Pipeline** from basic matrix operations using NumPy.


## Features Fully Implemented

### 1. First-Order Gradient Detectors

- Prewitt Operator: Computes differences using standard horizontal and vertical weight tiles.

- Sobel Operator: Employs weighted spatial kernels to smooth high-frequency local noise while computing gradients.

- Scharr Operator: Utilizes optimized central pixel coefficients for high-fidelity rotational symmetry processing.


### 2. Second-Order Derivative Detectors

- Laplacian Matrix: Highlights rapid spatial variation via computing secondary directional derivatives.

- Laplacian-of-Gaussian (LoG): Pre-filters with symmetric Gaussian distributions to suppress noise before computing second derivatives.


### 3. Step-by-Step Canny Pipeline

- Gaussian Smoothing: Blurs the target image matrix to eliminate microscopic structural variations.

- Sobel Matrix Projections: Computes spatial coordinate derivatives (G_x, G_y), gradient magnitudes, and exact continuous orientation matrices.

- Non-Maximum Suppression (NMS): Evaluates local orthogonal neighbor sectors (0°, 45°, 90°, 135°) to trim structural edge widths down to a clean 1-pixel baseline.

- Dual Matrix Thresholding: Maps weak/strong candidate edge coordinates based on user ratios.

- Hysteresis Tracking: Traverses adjacent 3 X 3 matrix neighborhoods to connect validated weak elements to confirmed strong structural anchors.


## Core Technologies & Tools

- Python
- OpenCV
- NumPy