import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- SCRATCH IMPLEMENTATIONS ---

def get_histogram_1d(img):
    """Calculates 1D histogram using NumPy."""
    return np.bincount(img.flatten(), minlength=256)

def global_histogram_equalization(img):
    """Implementation of GHE from scratch."""
    hist = get_histogram_1d(img)
    cdf = hist.cumsum()
    # Normalize mapping to 0-255
    cdf_m = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    return cdf_m[img].astype('uint8')

def histogram_matching(source, reference):
    """Matches the histogram of source to the reference."""
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(reference, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64) / source.size
    r_quantiles = np.cumsum(r_counts).astype(np.float64) / reference.size

    # Map source quantiles to reference quantiles
    interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)
    return interp_r_values[bin_idx].reshape(source.shape).astype('uint8')

def scratch_clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    """
    Simplified Adaptive Histogram Equalization.
    In a full implementation, bilinear interpolation is used between tiles.
    """
    h, w = img.shape
    th, tw = h // grid_size[0], w // grid_size[1]
    out = np.zeros_like(img)

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Extract tile
            tile = img[i*th:(i+1)*th, j*tw:(j+1)*tw]
            # Equalize tile
            hist = np.bincount(tile.flatten(), minlength=256)
            
            # Clip Limit (Redistribution)
            upper_limit = clip_limit * (tile.size / 256)
            excess = np.sum(np.maximum(hist - upper_limit, 0))
            hist = np.minimum(hist, upper_limit) + (excess / 256)
            
            cdf = hist.cumsum()
            cdf_m = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-6)
            out[i*th:(i+1)*th, j*tw:(j+1)*tw] = cdf_m[tile].astype('uint8')
    return out

def plot_hs_histogram(img_path):
    """Calculates and plots 2D H-S Histogram for a color image."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return
    
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist, _, _ = np.histogram2d(
        hsv[..., 0].flatten(), hsv[..., 1].flatten(), 
        bins=[180, 256], range=[[0, 180], [0, 256]]
    )
    plt.figure()
    plt.imshow(hist.T, interpolation='nearest', origin='lower', cmap='hot')
    plt.title('2D H-S Histogram (Color Info)')
    plt.colorbar(label='Pixel Count')
    plt.xlabel('Hue')
    plt.ylabel('Saturation')

# --- MAIN EXECUTION ---

def main():
    # File Names (Update these to your local file names)
    forest_file = 'dark forest.webp'
    xray_file = 'x-ray.webp'

    # Load Images
    img_forest = cv2.imread(forest_file, cv2.IMREAD_GRAYSCALE)
    img_xray = cv2.imread(xray_file, cv2.IMREAD_GRAYSCALE)

    if img_forest is None or img_xray is None:
        print("Error: Check your image filenames!")
        return

    # 1. Test GHE on Forest
    ghe_forest = global_histogram_equalization(img_forest)

    # 2. Test CLAHE on X-ray
    clahe_xray = scratch_clahe(img_xray)
    cv2_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img_xray)

    # 3. Test Matching (Source: Forest, Reference: X-ray)
    matched_img = histogram_matching(img_forest, img_xray)

    # --- VISUALIZATION ---
    
    # Task: Comparison of GHE vs Original
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(img_forest, cmap='gray'), plt.title('Original Forest')
    plt.subplot(132), plt.imshow(ghe_forest, cmap='gray'), plt.title('GHE Result')
    plt.subplot(133), plt.hist(img_forest.ravel(), 256, [0,256], color='r', alpha=0.5), \
                      plt.hist(ghe_forest.ravel(), 256, [0,256], color='b', alpha=0.5), \
                      plt.title('1D Histogram Shift')
    plt.show()

    # Task: CLAHE vs X-ray
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(img_xray, cmap='gray'), plt.title('Original X-ray')
    plt.subplot(132), plt.imshow(clahe_xray, cmap='gray'), plt.title('Scratch CLAHE')
    plt.subplot(133), plt.imshow(cv2_clahe, cmap='gray'), plt.title('OpenCV CLAHE')
    plt.show()

    # Task: Histogram Matching
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(img_forest, cmap='gray'), plt.title('Source (Forest)')
    plt.subplot(122), plt.imshow(matched_img, cmap='gray'), plt.title('Matched to X-ray Profile')
    plt.show()

    # Task: 2D Histogram (using original forest color file)
    plot_hs_histogram(forest_file)
    plt.show()

if __name__ == "__main__":
    main()