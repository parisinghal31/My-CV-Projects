import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_receipt():
    """
    Generates a synthetic crumpled receipt image with text, 
    uneven illumination gradients, and sharp shadow artifacts.
    """
    # 1. Create a clean white canvas
    height, width = 600, 500
    img = np.ones((height, width), dtype=np.uint8) * 240  # Off-white paper base
    
    # 2. Add structural document text
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX]
    receipt_text = [
        ("COFFEE SHOP #4221", 1.0, 40, 2),
        ("-----------------------------", 0.8, 75, 1),
        ("1x ESPRESSO        $3.50", 0.7, 120, 1),
        ("1x AVOCADO TOAST   $8.50", 0.7, 160, 1),
        ("2x CROISSANT       $6.00", 0.7, 200, 1),
        ("-----------------------------", 0.8, 240, 1),
        ("SUBTOTAL          $18.00", 0.7, 280, 1),
        ("TAX (8.25%)        $1.49", 0.7, 320, 1),
        ("TOTAL             $19.49", 0.8, 370, 2),
        ("-----------------------------", 0.8, 410, 1),
        ("THANK YOU FOR YOUR VISIT!", 0.6, 460, 1),
        ("STORE ID: X77-991A", 0.5, 500, 1)
    ]
    
    for text, scale, y_pos, thickness in receipt_text:
        # Center the text horizontally
        text_size = cv2.getTextSize(text, fonts[0], scale, thickness)[0]
        x_pos = (width - text_size[0]) // 2
        cv2.putText(img, text, (x_pos, y_pos), fonts[0], scale, (40), thickness, cv2.LINE_AA)
        
    # 3. Simulate Paper Wrinkles / Creases
    # Generate high-frequency noise and use distance transform to simulate ridge structures
    noise = np.zeros((height, width), dtype=np.uint8)
    cv2.randu(noise, 0, 255)
    _, noise_thresh = cv2.threshold(noise, 252, 255, cv2.THRESH_BINARY)
    dist_transform = cv2.distanceTransform(255 - noise_thresh, cv2.DIST_L2, 3)
    cv2.normalize(dist_transform, dist_transform, 0, 15, cv2.NORM_MINMAX)
    img = cv2.subtract(img, dist_transform.astype(np.uint8))

    # 4. Apply Global Illumination Gradient (Uneven Light)
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    # Light source shining from top-left, fading out heavily to bottom-right
    gradient = 1.0 - (X / (width * 1.5) + Y / (height * 1.5))
    gradient = np.clip(gradient, 0.3, 1.0)
    img = (img * gradient).astype(np.uint8)
    
    # 5. Apply a Harsh Shadow Overlay (simulate a hand/phone blocking light)
    # Create a sharp diagonal mask with a slightly soft blur edge
    shadow_mask = np.ones((height, width), dtype=np.float32)
    # Equation of line for shadow edge: Y = X + 100
    for y in range(height):
        for x in range(width):
            if y > (x + 50):
                shadow_mask[y, x] = 0.45  # Dim the pixels significantly in the shadow zone
    
    shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
    img = np.clip(img * shadow_mask, 0, 255).astype(np.uint8)
    
    return img

def sauvola_threshold(gray_img, window_size=25, k=0.2, R=128):
    """
    Applies Sauvola local adaptive thresholding using Integral Images for O(1) efficiency.
    """
    if window_size % 2 == 0:
        window_size += 1
    
    half_w = window_size // 2
    
    # Pad image to handle borders
    padded = cv2.copyMakeBorder(gray_img, half_w, half_w, half_w, half_w, cv2.BORDER_REPLICATE)
    padded = padded.astype(np.float64)
    
    # Calculate Integral Images
    sum_int = cv2.integral(padded)
    sqsum_int = cv2.integral(padded ** 2)
    
    # Layout meshgrids for coordinate evaluations
    y1 = np.arange(0, gray_img.shape[0])
    y2 = y1 + window_size
    x1 = np.arange(0, gray_img.shape[1])
    x2 = x1 + window_size
    
    X1, Y1 = np.meshgrid(x1, y1)
    X2, Y2 = np.meshgrid(x2, y2)
    
    area = window_size * window_size
    
    # Sum windows
    local_sum = sum_int[Y2, X2] - sum_int[Y1, X2] - sum_int[Y2, X1] + sum_int[Y1, X1]
    mean = local_sum / area
    
    # Square sum windows
    local_sq_sum = sqsum_int[Y2, X2] - sqsum_int[Y1, X2] - sqsum_int[Y2, X1] + sqsum_int[Y1, X1]
    variance = (local_sq_sum / area) - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 0))
    
    # Sauvola Formula evaluation
    threshold = mean * (1.0 + k * ((std / R) - 1.0))
    
    return np.where(gray_img > threshold, 255, 0).astype(np.uint8)

def execute_pipeline():
    # 1. Self-generate the corrupted sample image
    print("Generating synthetic crumpled receipt with complex illumination profiles...")
    img = generate_synthetic_receipt()
    
    # 2. Global Otsu Execution
    _, t_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Adaptive Mean Execution
    t_adapt_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 25, 10)
    
    # 4. Adaptive Gaussian Execution
    t_adapt_gauss = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 25, 10)
    
    # 5. Native Sauvola Implementation Engine
    t_sauvola = sauvola_threshold(img, window_size=25, k=0.18)
    
    # Plotting and mapping the results matrix
    titles = ['Generated Asset', "Global Otsu's", 'Adaptive Mean', 'Adaptive Gaussian', 'Sauvola (Custom)']
    images = [img, t_otsu, t_adapt_mean, t_adapt_gauss, t_sauvola]
    
    print("Processing algorithms and building visualization plots...")
    plt.figure(figsize=(15, 9))
    for i in range(5):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=12, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    output_filename = 'binarization_benchmark.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Success! Benchmark output saved safely to '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    execute_pipeline()