import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_nl_means, estimate_sigma

def load_and_preprocess(image_path):
    """Loads an image in grayscale and normalizes it to [0, 1] float range."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not open or find the image at: {image_path}")
    return img.astype(np.float32) / 255.0

def add_gaussian_noise(img, mean=0.0, sigma=0.1):
    noise = np.random.normal(mean, sigma, img.shape)
    noisy_img = img + noise
    return np.clip(noisy_img, 0.0, 1.0)

def add_salt_and_pepper_noise(img, amount=0.05):
    noisy_img = np.copy(img)
    # Salt (white pixels)
    num_salt = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_img[tuple(coords)] = 1.0
    
    # Pepper (black pixels)
    num_pepper = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_img[tuple(coords)] = 0.0
    return noisy_img

def add_poisson_noise(img):
    # Poisson noise is signal dependent. Scale up to simulate photon counts.
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_img = np.random.poisson(img * vals) / float(vals)
    return np.clip(noisy_img, 0.0, 1.0)

def apply_filters(noisy_img):
    """Applies the 5 target restoration filters."""
    # Convert back to uint8 safely for filters that require it
    noisy_u8 = (noisy_img * 255).astype(np.uint8)
    results = {}
    
    # 1. Mean Filter (Box Filter)
    mean_f = cv2.blur(noisy_img, (5, 5))
    results['Mean'] = np.clip(mean_f, 0.0, 1.0)
    
    # 2. Median Filter
    median_f = cv2.medianBlur(noisy_u8, 5)
    results['Median'] = median_f.astype(np.float32) / 255.0
    
    # 3. Gaussian Filter
    gaussian_f = cv2.GaussianBlur(noisy_img, (5, 5), sigmaX=1.0)
    results['Gaussian'] = np.clip(gaussian_f, 0.0, 1.0)
    
    # 4. Bilateral Filter
    # Needs uint8 input for standard opencv implementation processing
    bilateral_f = cv2.bilateralFilter(noisy_u8, d=9, sigmaColor=50, sigmaSpace=15)
    results['Bilateral'] = bilateral_f.astype(np.float32) / 255.0
    
    # 5. Non-Local Means Filter (NLM)
    sigma_est = np.mean(estimate_sigma(noisy_img, channel_axis=None))
    nlm_f = denoise_nl_means(noisy_img, h=1.15 * sigma_est, fast_mode=True, 
                             patch_size=5, patch_distance=6, channel_axis=None)
    results['NL-Means'] = np.clip(nlm_f, 0.0, 1.0)
    
    return results

def run_benchmark(image_path, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    orig = load_and_preprocess(image_path)
    
    noises = {
        'Gaussian': add_gaussian_noise(orig, sigma=0.1),
        'Salt & Pepper': add_salt_and_pepper_noise(orig, amount=0.04),
        'Poisson': add_poisson_noise(orig)
    }
    
    print(f"{'Noise Type':<15} | {'Filter':<12} | {'PSNR (dB)':<10} | {'SSIM':<8}")
    print("-" * 55)
    
    fig, axes = plt.subplots(3, 7, figsize=(20, 10))
    
    for row_idx, (noise_name, noisy_img) in enumerate(noises.items()):
        # Metrics for baseline noisy image
        p_noisy = psnr(orig, noisy_img, data_range=1.0)
        s_noisy = ssim(orig, noisy_img, data_range=1.0)
        
        # Display baseline
        axes[row_idx, 0].imshow(orig, cmap='gray')
        axes[row_idx, 0].set_title("Original Clean")
        axes[row_idx, 1].imshow(noisy_img, cmap='gray')
        axes[row_idx, 1].set_title(f"{noise_name}\nPSNR: {p_noisy:.2f}")
        
        # Process and evaluate filters
        filter_outputs = apply_filters(noisy_img)
        
        for col_idx, (filter_name, restored_img) in enumerate(filter_outputs.items(), start=2):
            p_score = psnr(orig, restored_img, data_range=1.0)
            s_score = ssim(orig, restored_img, data_range=1.0)
            
            print(f"{noise_name:<15} | {filter_name:<12} | {p_score:<10.2f} | {s_score:<8.4f}")
            
            ax = axes[row_idx, col_idx]
            ax.imshow(restored_img, cmap='gray')
            ax.set_title(f"{filter_name}\nPSNR: {p_score:.2f}\nSSIM: {s_score:.3f}", fontsize=9)
            
    for ax in axes.ravel():
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_matrix.png'), dpi=300)
    plt.show()
    print(f"\nBenchmark finished. Visual matrix saved to '{output_dir}/benchmark_matrix.png'")

if __name__ == '__main__':
    # Ensure a target sample image exists or replace with your local file path
    sample_img_name = 'sample_clean.png'
    if not os.path.exists(sample_img_name):
        # Generate a synthetic template image if no input file is found
        synthetic_clean = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(synthetic_clean, (40, 40), (216, 216), 180, -1)
        cv2.circle(synthetic_clean, (128, 128), 50, 255, -1)
        cv2.putText(synthetic_clean, "TEST", (90, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        cv2.imwrite(sample_img_name, synthetic_clean)
        print(f"Generated synthetic baseline file: '{sample_img_name}'")
        
    run_benchmark(sample_img_name)