# Shape Sorter using Classical Computer Vision

## Project Overview

Shape Sorter is a Computer Vision project that detects, classifies, counts, and visualizes geometric shapes present in an image. The system uses classical image processing techniques and does not rely on Machine Learning algorithms.

The project identifies the following shapes:

* Triangle
* Square
* Rectangle
* Pentagon
* Hexagon
* Circle
* Oval (Ellipse)

In addition to classification, the project extracts Hu Moments for each detected shape, providing invariant shape descriptors that remain stable under translation, scaling, and rotation.

---

## Objective

The objective of this project is to:

* Detect multiple shapes from a single image.
* Classify each shape accurately.
* Count the number of occurrences of each shape.
* Visualize detections using contours and bounding boxes.
* Extract Hu Moments for shape analysis.
* Demonstrate shape recognition using only traditional Computer Vision techniques.

---

## Features

### Shape Detection

Detects geometric shapes from a colored image containing multiple objects.

### Shape Classification

Classifies detected objects into:

* Triangle
* Square
* Rectangle
* Pentagon
* Hexagon
* Circle
* Oval

### Shape Counting

Maintains a count of each detected shape.

### Visualization

Draws:

* Green contour around each detected shape.
* Blue bounding box.
* Red shape label.

### Hu Moments Extraction

Computes seven Hu Moments for every detected contour.

### No Machine Learning

The entire pipeline is built using:

* Contours
* Polygon Approximation
* Geometric Analysis
* Hu Moments


---

# Methodology

The complete pipeline is shown below:

Input Image

↓

HSV Color Segmentation

↓

Morphological Processing

↓

Contour Detection

↓

Polygon Approximation (approxPolyDP)

↓

Shape Classification

↓

Hu Moment Extraction

↓

Shape Counting

↓

Visualization & Output

---

# Why HSV Color Segmentation?

The input image contains highly saturated colored shapes on a light gray background.

Examples:

* Green Square
* Cyan Circle
* Magenta Triangle
* Yellow Pentagon
* Orange Oval
* Red Rectangle

In grayscale, some colors (especially yellow) become very similar to the background, making contour extraction difficult.

To overcome this issue, the image is converted from BGR to HSV color space.

HSV allows us to separate:

* High-saturation colored objects
* Low-saturation gray background

This produces a clean binary mask that isolates the shapes.

---

# Shape Detection

After segmentation:

```python
contours, _ = cv2.findContours(...)
```

OpenCV extracts the boundary of every shape.

Each contour is processed independently.

---

# Polygon Approximation

The project uses:

```python
cv2.approxPolyDP()
```

which implements the Douglas-Peucker algorithm.

The algorithm simplifies a contour into a polygon with fewer vertices.

Example:

Raw contour:

120 points

↓

Polygon approximation

↓

5 points

↓

Pentagon

This greatly simplifies shape classification.

---

# Shape Classification Logic

## Triangle

If the approximated contour contains:

```text
3 vertices
```

the shape is classified as a Triangle.

---

## Square

If the contour contains:

```text
4 vertices
```

the aspect ratio is computed:

```text
width / height
```

If:

```text
0.95 ≤ aspect ratio ≤ 1.05
```

the shape is classified as a Square.

---

## Rectangle

If:

```text
vertices = 4
```

and the aspect ratio is not close to 1, the shape is classified as a Rectangle.

---

## Pentagon

If:

```text
vertices = 5
```

the shape is classified as a Pentagon.

---

## Hexagon

If:

```text
vertices = 6
```

the shape is classified as a Hexagon.

---

## Circle and Oval

For contours with more than six vertices, geometric properties are analyzed.

### Circularity

Circularity is computed as:

```text
4π × Area / Perimeter²
```

A perfect circle has circularity close to:

```text
1
```

### Ellipse Fitting

OpenCV fits an ellipse to the contour:

```python
cv2.fitEllipse()
```

The ratio between the major and minor axes is computed.

#### Circle

If:

* Circularity is high
* Axis ratio is close to 1

the contour is classified as a Circle.

#### Oval

If:

* Circularity is lower
* Axis ratio differs significantly from 1

the contour is classified as an Oval.

---

# Hu Moments

The project extracts seven Hu Moments for every contour.

Implementation:

```python
moments = cv2.moments(contour)
hu = cv2.HuMoments(moments)
```

Hu Moments are:

* Translation invariant
* Rotation invariant
* Scale invariant

These descriptors uniquely characterize the geometry of a shape.

Example output:

```text
Shape : Pentagon

Hu Moments:
[0.73 2.11 5.78 7.92 10.00 9.65 10.00]
```

---

# Technologies Used

## Programming Language

* Python 3.x

## Libraries

* OpenCV
* NumPy
* Collections

---


# Project Structure

```text
Shape-Sorter/

│
├── main.py
│
├── SampleInputImage.jpg
│
├── Detected_Shapes_Output.jpg
│
├── README.md
```

---


# Sample Output

```text
============================================================
DETECTED SHAPES
============================================================

Shape : Square
Vertices : 4

Shape : Circle
Vertices : 8

Shape : Triangle
Vertices : 3

Shape : Pentagon
Vertices : 5

Shape : Oval
Vertices : 8

Shape : Rectangle
Vertices : 4

============================================================
SHAPE COUNTS
============================================================

Circle     : 1
Oval       : 1
Pentagon   : 1
Rectangle  : 1
Square     : 1
Triangle   : 1

Total Shapes : 6
```

---


# Applications

* Industrial object sorting
* Quality inspection systems
* Shape-based object recognition
* Educational Computer Vision projects
* Automated manufacturing systems
* Robotics and pick-and-place applications

---


# Conclusion

This project demonstrates how geometric shapes can be detected and classified using only classical Computer Vision techniques. By combining HSV color segmentation, contour extraction, polygon approximation, geometric analysis, and Hu Moments, the system accurately identifies and counts multiple shapes without relying on Machine Learning.

The project serves as a strong example of traditional image processing methods and provides a foundation for more advanced object recognition systems.
