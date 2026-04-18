# Histogram Lab: Image Enhancement from Scratch
- A comprehensive study of image intensity distributions, implementing classical enhancement algorithms using NumPy and verifying results against OpenCV.

# Features:
- 1D/2D Histograms: Statistical analysis of pixel intensities and H-S color relationships.
- Global Histogram Equalization (GHE): Contrast stretching for underexposed photography.
- CLAHE: Contrast Limited Adaptive Histogram Equalization for medical X-ray detail preservation.
- Histogram Matching: Transferring the visual profile of a reference image to a target image.

# Mathematical Foundations:

## Global Histogram Equalization
The transformation function $T(r_k)$ is based on the Cumulative Distribution Function (CDF):
s_k = \sum_{j=0}^{k} \frac{n_j}{MN}
where MN is the total number of pixels. This maps the input probability density to a uniform distribution.

## Histogram Matching
Matching solves the problem of finding a mapping z = G^{-1}(T(r)) such that the output matches a specific target PDF. This is implemented via a lookup table (LUT) generated from the CDFs of both images.


# Results & Verification:
- The implementation is verified by calculating the Mean Squared Error (MSE) between the custom NumPy implementation and `cv2.equalizeHist()`. 

| Dataset | Method | Result |
| :--- | :--- | :--- |
| Underexposed | GHE | Significant detail recovery in shadows. |
| Chest X-ray | CLAHE | Enhanced rib-cage/lung contrast without noise amplification. |

# Tools used:
- Python 
- NumPy
- OpenCV (for verification)
- Matplotlib (for visualization)