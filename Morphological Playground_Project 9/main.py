import numpy as np
import cv2
import os

def create_synthetic_fingerprint():
    """Generates a synthetic noisy fingerprint mask pattern."""
    img = np.zeros((300, 300), dtype=np.uint8)
    # Draw concentric ridge arcs
    for r in range(40, 140, 14):
        cv2.circle(img, (150, 180), r, 255, 3)
    
    # Inject pepper noise (breaks lines)
    pepper_mask = np.random.choice([0, 1], size=img.shape, p=[0.05, 0.95]).astype(np.uint8)
    img = img * pepper_mask
    
    # Inject salt noise (bright spots in background)
    salt_mask = (np.random.choice([0, 1], size=img.shape, p=[0.98, 0.02]) * 255).astype(np.uint8)
    img = cv2.bitwise_or(img, salt_mask)
    return img

def create_synthetic_car_plate():
    """Generates a synthetic vehicle profile frame with a high-contrast license plate."""
    img = np.ones((400, 600), dtype=np.uint8) * 80 # Dark gray car body
    
    # Create background lighting gradient (vignetting/glare simulation)
    for i in range(400):
        img[i, :] = np.clip(img[i, :] + (i // 6), 0, 255)
        
    # Draw white license plate background box
    cv2.rectangle(img, (200, 230), (400, 290), 220, -1)
    
    # Draw high-contrast text characters inside the plate box
    cv2.putText(img, "MH14-AA-1234", (215, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 20, 2, cv2.LINE_AA)
    
    # Add random unrelated visual noise clutter (bumper lines)
    cv2.line(img, (50, 320), (550, 320), 30, 4)
    return img

if __name__ == '__main__':
    output_dir = "morphology_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Phase 1: Fingerprint Enhancement & Ridge Extraction ---")
    fingerprint = create_synthetic_fingerprint()
    
    # Define Structuring Elements (Kernels)
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    
    # Execute structural filters step-by-step
    eroded = cv2.erode(fingerprint, kernel_cross)
    dilated = cv2.dilate(fingerprint, kernel_cross)
    opened = cv2.morphologyEx(fingerprint, cv2.MORPH_OPEN, kernel_ellipse)
    closed = cv2.morphologyEx(fingerprint, cv2.MORPH_CLOSE, kernel_ellipse)
    
    # Unified clean sequence: Fusing broken gaps first, then brushing out background noise
    clean_fingerprint = cv2.morphologyEx(fingerprint, cv2.MORPH_CLOSE, kernel_ellipse)
    clean_fingerprint = cv2.morphologyEx(clean_fingerprint, cv2.MORPH_OPEN, kernel_ellipse)
    
    # Extractions
    gradient = cv2.morphologyEx(fingerprint, cv2.MORPH_GRADIENT, kernel_cross)
    top_hat = cv2.morphologyEx(fingerprint, cv2.MORPH_TOPHAT, kernel_ellipse)
    black_hat = cv2.morphologyEx(fingerprint, cv2.MORPH_BLACKHAT, kernel_ellipse)
    
    # Save fingerprint collection
    cv2.imwrite(f"{output_dir}/fp_0_original.png", fingerprint)
    cv2.imwrite(f"{output_dir}/fp_1_eroded.png", eroded)
    cv2.imwrite(f"{output_dir}/fp_2_dilated.png", dilated)
    cv2.imwrite(f"{output_dir}/fp_3_opened.png", opened)
    cv2.imwrite(f"{output_dir}/fp_4_closed.png", closed)
    cv2.imwrite(f"{output_dir}/fp_5_clean_ridges.png", clean_fingerprint)
    cv2.imwrite(f"{output_dir}/fp_6_gradient.png", gradient)
    
    print("--- Phase 2: License Plate Structural Pre-Processor ---")
    car_scene = create_synthetic_car_plate()
    
    # Step 1: Horizontal Top-Hat to isolate characters while killing illumination sweep
    plate_kernel_wide = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    th_car = cv2.morphologyEx(car_scene, cv2.MORPH_TOPHAT, plate_kernel_wide)
    
    # Step 2: Adaptive Thresholding
    _, thresh_car = cv2.threshold(th_car, 50, 255, cv2.THRESH_BINARY)
    
    # Step 3: Close character gaps horizontally to fuse text into one unified block
    close_kernel_box = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    plate_mask = cv2.morphologyEx(thresh_car, cv2.MORPH_CLOSE, close_kernel_box)
    
    # Step 4: Isolate ROI via image masking multiplication
    segmented_plate = cv2.bitwise_and(car_scene, car_scene, mask=plate_mask)
    
    # Save license plate stages
    cv2.imwrite(f"{output_dir}/lp_0_car_scene.png", car_scene)
    cv2.imwrite(f"{output_dir}/lp_1_tophat_isolated.png", th_car)
    cv2.imwrite(f"{output_dir}/lp_2_thresholded.png", thresh_car)
    cv2.imwrite(f"{output_dir}/lp_3_bounding_mask.png", plate_mask)
    cv2.imwrite(f"{output_dir}/lp_4_segmented_plate.png", segmented_plate)
    
    print(f"🎉 Success! All execution sequences complete. Asset files stored in './{output_dir}' folder.")
    
    # Interactive display loop
    cv2.imshow("FP Original", fingerprint)
    cv2.imshow("FP Extracted Clean Ridges", clean_fingerprint)
    cv2.imshow("FP Structural Gradient", gradient)
    cv2.imshow("LP Car Source Input", car_scene)
    cv2.imshow("LP Character Bounding Mask", plate_mask)
    cv2.imshow("LP Isolated Final Region", segmented_plate)
    
    print("\nDisplaying windows. Click 'X' on any image window or press 'q' to exit safely.")
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty("FP Original", cv2.WND_PROP_VISIBLE) < 1:
            break
            
    cv2.destroyAllWindows()