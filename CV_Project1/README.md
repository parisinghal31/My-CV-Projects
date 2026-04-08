# Project [1]: [Pixel Inspector project]
- Load an image, let the user click anywhere, and print the (B,G,R), (H,S,V), (L,a,b), and grayscale values
at that point. Draw a live 11×11 neighborhood zoom.

## Goal
- Internalize that an image is just numpy with channels and that BGR != RGB in OpenCV.

## Tools Used
- Python
- OpenCV
- Numpy

## Key Takeaway
- Coordinate [0,0] = top-left corner
- OpenCV uses BGR instead of RGB
- Images are 3d numpy arrays