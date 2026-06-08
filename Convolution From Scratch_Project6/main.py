import numpy as np
import cv2
import time
import os

def convolve2D(image, kernel, padding=0, stride=1):
    """
    Performs 2D convolution on grayscale or multi-channel images using pure NumPy.
    
    Parameters:
        image (numpy.ndarray): Input image array of shape (H, W) or (H, W, C).
        kernel (numpy.ndarray): 2D Kernel array of shape (Kh, Kw).
        padding (int): Number of zero-padded pixel layers to add around borders.
        stride (int): Pixel step size for the sliding window.
    """
    # Handle dimensions for Multi-Channel Color Images (H, W, Channels)
    if len(image.shape) == 3:
        h_in, w_in, c_in = image.shape
        # Process each color layer independently and stack them back together
        channels = [convolve2D(image[:, :, c], kernel, padding, stride) for c in range(c_in)]
        return np.stack(channels, axis=-1)
    
    # Handle Grayscale / Single-Channel Dimensions
    h_in, w_in = image.shape
    h_k, w_k = kernel.shape
    
    # Calculate explicit output dimensions using the CNN spatial sizing formula
    h_out = int((h_in - h_k + 2 * padding) / stride) + 1
    w_out = int((w_in - w_k + 2 * padding) / stride) + 1
    
    # Initialize the output matrix grid
    output = np.zeros((h_out, w_out))
    
    # Apply Zero Padding if requested
    if padding > 0:
        padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    else:
        padded_image = image
        
    # Sliding Window Extraction Loop
    for i in range(0, h_out):
        for j in range(0, w_out):
            start_i = i * stride
            end_i = start_i + h_k
            start_j = j * stride
            end_j = start_j + w_k
            
            # Slice the current patch and perform element-wise matrix multiplication & summation
            image_patch = padded_image[start_i:end_i, start_j:end_j]
            output[i, j] = np.sum(image_patch * kernel)
            
    return output

# --- Handcrafted Analytical Convolution Matrices ---
KERNELS = {
    "box_blur": np.ones((3, 3), dtype=np.float32) / 9.0,
    
    "gaussian_blur": np.array([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]], dtype=np.float32) / 16.0,
                               
    "sharpen": np.array([[ 0, -1,  0],
                         [-1,  5, -1],
                         [ 0, -1,  0]], dtype=np.float32),
                         
    "emboss": np.array([[-2, -1,  0],
                        [-1,  1,  1],
                        [ 0,  1,  2]], dtype=np.float32),
                        
    "sobel_horizontal_edge": np.array([[-1,  0,  1],
                                        [-2,  0,  2],
                                        [-1,  0,  1]], dtype=np.float32)
}

def run_pipeline(image_path="sample_input.jpg"):
    # 1. Create a clean sample input image if one does not exist
    if not os.path.exists(image_path):
        print(f"Creating a default synthetic image asset: '{image_path}'...")
        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        # Draw explicit shapes to give filters crisp edges to calculate gradients on
        cv2.rectangle(canvas, (50, 50), (200, 200), (0, 255, 0), -1)
        cv2.circle(canvas, (280, 280), 60, (0, 0, 255), -1)
        cv2.putText(canvas, "CNN 2D", (70, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.imwrite(image_path, canvas)

    # 2. Load the input source image
    src_img = cv2.imread(image_path)
    print(f"Successfully loaded '{image_path}' with shape: {src_img.shape}")
    
    # 3. CRITICAL: Cast to float32 before math calculations to prevent underflow wrap-around bugs
    src_img_float = src_img.astype(np.float32)
    
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Running Kernel Pipeline vs OpenCV Benchmark ---")
    
    for name, kernel in KERNELS.items():
        # Time the custom NumPy implementation
        t0 = time.time()
        custom_raw = convolve2D(src_img_float, kernel, padding=1, stride=1)
        # Safely clip out-of-bound floats [0, 255] and cast back to standard 8-bit image matrices
        custom_final = np.clip(custom_raw, 0, 255).astype(np.uint8)
        custom_time = time.time() - t0
        
        # Time the production OpenCV engine
        t1 = time.time()
        opencv_final = cv2.filter2D(src_img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        opencv_time = time.time() - t1
        
        # Calculate Mean Absolute Difference to verify mathematical correctness
        mad = np.mean(np.abs(custom_final.astype(np.float32) - opencv_final.astype(np.float32)))
        
        print(f"[{name.upper()}]")
        print(f"  -> NumPy Time:  {custom_time:.4f}s")
        print(f"  -> OpenCV Time: {opencv_time:.4f}s (Speed Ratio: {custom_time/opencv_time:.1f}x)")
        print(f"  -> Mathematical Variance (M.A.D.): {mad:.4f}")
        
        # Save output image
        out_path = os.path.join(output_dir, f"output_{name}.jpg")
        cv2.imwrite(out_path, custom_final)
        print(f"  -> Saved output to: {out_path}\n")

if __name__ == "__main__":
    run_pipeline()