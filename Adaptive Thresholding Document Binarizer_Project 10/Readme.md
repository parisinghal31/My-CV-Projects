# Adaptive Thresholding Document Binarizer

- An advanced, scanner-grade document binarization engine built to process highly degraded, crumpled, and unevenly illuminated document images (e.g., receipts, historical manuscripts). 

- This project implements and evaluates four critical segmentation methodologies, featuring a custom vectorized **Sauvola Binarization** architecture optimized via **Integral Images (Summed-Area Tables)**.


##  Features

* **Comparative Framework:** Evaluate Global Otsu vs. Local Adaptive methods simultaneously.

* **Vectorized Sauvola Engine:** High-performance, native NumPy implementation of Sauvola’s algorithm running in $O(1)$ algorithmic time complexity relative to window size.

* **Illumination Invariance:** Resilient to casting shadows, geometric folds, low contrast, and gradient light sources.


##  Algorithm Insights

### 1. Global Otsu's Thresholding

- Finds a singular global threshold value that maximizes inter-class variance between foreground text and background. Completely breaks down under uneven lighting scenarios because it lacks local spatial awareness.


### 2. Adaptive Mean & Gaussian

- Calculates a distinct threshold for every pixel based on an $N \times N$ sliding window. 

* **Mean:** Simple local average. Tends to generate blocky artifacts along sharp shadow gradients.

* **Gaussian:** Threshold is a weighted sum using a Gaussian distribution, producing cleaner text margins but occasionally dropping low-contrast character lines.


### 3. Sauvola's Thresholding (Custom Implementation)

- Tailored explicitly for textual documents. It dynamically calculates thresholds based on both the local mean ($\mu$) and standard deviation ($\sigma$):

$$T(x, y) = \mu(x, y) \cdot \left[ 1 + k \cdot \left( \frac{\sigma(x, y)}{R} - 1 \right) \right]$$

Where:
* $R$ is the dynamic range of standard deviation (fixed at 128 for 8-bit images).
* $k$ is a user-defined scaling parameter (default: `0.2`) controlling the text edge margin hardness.


##  Project Structure

├── main.py                     # Main source execution script
├── binarization_benchmark.png # Benchmark artifact output
└── Readme.md