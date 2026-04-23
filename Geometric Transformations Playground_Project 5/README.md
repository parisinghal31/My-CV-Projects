Project 5: Geometric Transformations Playground

- This project explores the mathematical foundations of computer vision by implementing 2D and 3D geometric transformations. Instead of relying solely on high-level functions, we manually construct transformation matrices to understand how pixels are mapped from a source image to a destination coordinate system.


# Transformation Types:

1. Affine Transformations (2 X 3 Matrix)
- Affine transformations preserve collinearity and ratios of distances. We implement:
Translation: Moving the image along the x and y axes.
Scaling: Resizing the image.
Rotation: Pivoting the image around a specific point.
Shearing: Shifting one part of the image, causing a "slant."

2. Perspective Transformations (3 X 3 Matrix)
- Unlike affine transforms, perspective transformations do not preserve parallelism. This allows us to simulate depth and rectify images taken at an angle.


# The Document Scanner:
- The capstone of this project is a Perspective Rectifier. It takes an image of a document photographed at a tilt and "unwarps" it into a flat, top-down view.
- Workflow:
 Define Source Points: Identifying the four corners of the paper in the original image.
 Define Destination Points: Defining the corners of a clean rectangle (e.g., A4 aspect ratio).
 Compute Homography: Calculating the 3 X 3 transformation matrix.
 Warp: Applying cv2.warpPerspective to generate the final "scan."


#Requirements:
- Python 3.x
- OpenCV (cv2)
- NumPy


#How to Run:

- Ensure you have the requirements installed: pip install opencv-python numpy.
- Run the script: python transformation_playground.py.
- The script will generate a checkerboard pattern and demonstrate each transformation in separate windows.