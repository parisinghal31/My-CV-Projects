import numpy as np
import cv2
import os

def convolution2d(image, kernel):
    """
    Performs 2D convolution with replication padding to maintain image size.
    """
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape
    
    pad_h = ker_h // 2
    pad_w = ker_w // 2
    
    # Use cv2 only for efficient padding allocation
    padded_img = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    output = np.zeros_like(image, dtype=np.float64)
    
    for i in range(img_h):
        for j in range(img_w):
            roi = padded_img[i:i+ker_h, j:j+ker_w]
            output[i, j] = np.sum(roi * kernel)
            
    return output

def get_kernels():
    return {
        'prewitt_x': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64),
        'prewitt_y': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64),
        
        'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64),
        'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64),
        
        'scharr_x': np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64),
        'scharr_y': np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float64),
        
        'laplacian': np.array([[0, 1, 0], [1, -4, 0], [0, 1, 0]], dtype=np.float64),
        
        'log': np.array([[0,  0, -1,  0,  0],
                          [0, -1, -2, -1,  0],
                          [-1, -2, 16, -2, -1],
                          [0, -1, -2, -1,  0],
                          [0,  0, -1,  0,  0]], dtype=np.float64)
    }

def first_order_detector(image, method='sobel'):
    kernels = get_kernels()
    kx = kernels[f'{method}_x']
    ky = kernels[f'{method}_y']
    
    gx = convolution2d(image, kx)
    gy = convolution2d(image, ky)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    else:
        magnitude = magnitude.astype(np.uint8)
    return magnitude, gx, gy

def laplacian_detector(image, method='laplacian'):
    kernel = get_kernels()[method]
    grad = convolution2d(image, kernel)
    grad = np.abs(grad)
    if grad.max() > 0:
        return (grad / grad.max() * 255).astype(np.uint8)
    return grad.astype(np.uint8)

def non_maximum_suppression(magnitude, gx, gy):
    h, w = magnitude.shape
    nms_out = np.zeros((h, w), dtype=np.float64)
    
    angle = np.arctan2(gy, gx) * 180.0 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q = 255
            r = 255
            curr_angle = angle[i, j]
            
            # Group continuous gradient angles into discrete sectors (0, 45, 90, 135)
            if (0 <= curr_angle < 22.5) or (157.5 <= curr_angle <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif (22.5 <= curr_angle < 67.5):
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif (67.5 <= curr_angle < 112.5):
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif (112.5 <= curr_angle < 157.5):
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]
                
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                nms_out[i, j] = magnitude[i, j]
            else:
                nms_out[i, j] = 0
                
    return nms_out

def double_threshold(image, low_threshold_ratio, high_threshold_ratio):
    high_threshold = image.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio
    
    h, w = image.shape
    res = np.zeros((h, w), dtype=np.uint8)
    
    weak = 75
    strong = 255
    
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return res, weak, strong

def edge_tracking_hysteresis(image, weak, strong=255):
    h, w = image.shape
    out = image.copy()
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if out[i, j] == weak:
                # Check the immediate 8-way neighborhood boundaries
                neighbor_block = out[i-1:i+2, j-1:j+2]
                if np.any(neighbor_block == strong):
                    out[i, j] = strong
                else:
                    out[i, j] = 0
    return out

def custom_canny(image, low_ratio=0.05, high_ratio=0.15, gaussian_kernel_size=5):
    # Phase 1: Noise Reduction via Gaussian Blur
    blurred = cv2.GaussianBlur(image, (gaussian_kernel_size, gaussian_kernel_size), 1.4)
    # Phase 2: Structural Gradients via Sobel Operator
    mag, gx, gy = first_order_detector(blurred, method='sobel')
    # Phase 3: Thinning via Non-Maximum Suppression
    nms = non_maximum_suppression(mag, gx, gy)
    # Phase 4: Segmentation via Double Thresholding
    thresholded, weak, strong = double_threshold(nms, low_ratio, high_ratio)
    # Phase 5: Structural Linkage via Hysteresis
    canny_output = edge_tracking_hysteresis(thresholded, weak, strong)
    return canny_output

def generate_synthetic_image():
    """Generates a clean synthetic shape with structural variations for testing."""
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(img, (150, 150), 65, 255, -1)
    cv2.rectangle(img, (50, 50), (250, 250), 255, 3)
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

if __name__ == '__main__':
    image_path = "assets/sample.jpg"
    
    if os.path.exists(image_path):
        print(f"Loading real image matrix from: {image_path}")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        print(f"Target path '{image_path}' not found. Generating noisy synthetic matrix baseline...")
        img = generate_synthetic_image()

    print("\n--- Computing Edge Suites from First Principles ---")
    prewitt = first_order_detector(img, method='prewitt')[0]
    sobel = first_order_detector(img, method='sobel')[0]
    scharr = first_order_detector(img, method='scharr')[0]
    laplacian = laplacian_detector(img, method='laplacian')
    log = laplacian_detector(img, method='log')
    
    print("Executing Step-by-Step Custom Canny Pipeline...")
    my_canny = custom_canny(img, low_ratio=0.06, high_ratio=0.18)
    
    print("Evaluating baseline against official OpenCV binary standard...")
    cv2_canny = cv2.Canny(img, 40, 120)

    discrepancy = np.mean((my_canny.astype(float) - cv2_canny.astype(float)) ** 2)
    print(f"Mean Squared Discrepancy Margin vs cv2.Canny: {discrepancy:.4f}")

    # File Writing Engine (Saves high-quality PNG tracking frames)
    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(f"{output_dir}/0_original.png", img)
    cv2.imwrite(f"{output_dir}/1_prewitt.png", prewitt)
    cv2.imwrite(f"{output_dir}/2_sobel.png", sobel)
    cv2.imwrite(f"{output_dir}/3_scharr.png", scharr)
    cv2.imwrite(f"{output_dir}/4_laplacian.png", laplacian)
    cv2.imwrite(f"{output_dir}/5_log.png", log)
    cv2.imwrite(f"{output_dir}/6_custom_canny.png", my_canny)
    cv2.imwrite(f"{output_dir}/7_opencv_canny.png", cv2_canny)
    
    print(f"\n🎉 Success! All 8 edge maps have been saved inside the '{output_dir}' folder.")

    # Interactive Display Windows
    cv2.imshow("0. Original Input / Synthetic Test Data", img)
    cv2.imshow("1. Prewitt Detector", prewitt)
    cv2.imshow("2. Sobel Detector", sobel)
    cv2.imshow("3. Scharr Matrix", scharr)
    cv2.imshow("4. Laplacian Variant", laplacian)
    cv2.imshow("5. Laplacian of Gaussian (LoG)", log)
    cv2.imshow("6. Custom Canny Pipeline (From Scratch)", my_canny)
    cv2.imshow("7. Reference OpenCV cv2.Canny", cv2_canny)
    
    print("\nDisplaying windows. Focus on any image window and press any key to terminate script.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()