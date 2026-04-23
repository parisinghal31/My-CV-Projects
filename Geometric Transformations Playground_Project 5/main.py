import cv2
import numpy as np

def manual_affine_transformations(image):
    rows, cols = image.shape[:2]

    # 1. Translation Matrix (2x3): Shift 100px Right, 50px Down
    # [1 0 tx]
    # [0 1 ty]
    M_translate = np.float32([[1, 0, 100], 
                              [0, 1, 50]])
    translation = cv2.warpAffine(image, M_translate, (cols, rows))

    # 2. Rotation Matrix (2x3): 45 degrees around center
    angle = np.radians(45)
    c, s = np.cos(angle), np.sin(angle)
    # Rotation about origin (0,0) and then translated to keep in frame
    M_rotate = np.float32([[c, -s, cols/4], 
                           [s,  c, 0]])
    rotation = cv2.warpAffine(image, M_rotate, (cols, rows))

    # 3. Scaling Matrix (2x3): 0.5x width, 0.7x height
    # [sx 0  0]
    # [0  sy 0]
    M_scale = np.float32([[0.5, 0, 0], 
                          [0, 0.7, 0]])
    scaling = cv2.warpAffine(image, M_scale, (cols, rows))

    # 4. Shearing Matrix (2x3): Horizontal shear
    # [1  shx 0]
    # [0  1   0]
    M_shear = np.float32([[1, 0.5, 0], 
                          [0, 1, 0]])
    shearing = cv2.warpAffine(image, M_shear, (cols, rows))

    return translation, rotation, scaling, shearing

def document_scanner(image):
    """
    Applies a 3x3 Perspective Warp to simulate a document scanner.
    """
    rows, cols = image.shape[:2]

    # Source points: These would typically be the 4 corners of a paper 
    # found in a phone photo (tilted perspective).
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    src_pts = np.float32([[150, 150], [cols-100, 100], 
                          [cols-50, rows-50], [50, rows-100]])

    # Destination points: The "flat" top-down view
    width, height = 400, 600
    dst_pts = np.float32([[0, 0], [width, 0], 
                          [width, height], [0, height]])

    # Manual creation of Perspective Matrix is complex (requires solving 8 equations), 
    # so we use getPerspectiveTransform for the 3x3 matrix.
    M_perspective = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Warp the perspective
    scanned = cv2.warpPerspective(image, M_perspective, (width, height))
    
    return scanned, src_pts

# Main Execution
if __name__ == "__main__":
    # Create a placeholder image (checkerboard) 
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    for i in range(0, 500, 50):
        for j in range(0, 500, 50):
            if (i+j) // 50 % 2 == 0:
                img[i:i+50, j:j+50] = (255, 255, 255)

    # Apply transformations
    trans, rot, scale, shear = manual_affine_transformations(img)
    scan, src_pts = document_scanner(img)

    # Show results
    cv2.imshow("Translation", trans)
    cv2.imshow("Rotation", rot)
    cv2.imshow("Shearing", shear)
    
    # Draw original "tilted" points on the original for visualization
    img_viz = img.copy()
    for pt in src_pts:
        cv2.circle(img_viz, tuple(pt.astype(int)), 10, (0, 0, 255), -1)
    
    cv2.imshow("Scanner Source Points", img_viz)
    cv2.imshow("Flat Scanned Result", scan)

    print("Geometric Playground Complete. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()