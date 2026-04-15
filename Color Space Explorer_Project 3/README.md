# Project 3: Color Space Explorer & HSV Segmenter

# Goal
- To build an interactive tool that explores different mathematical representations of color (Color Spaces) and implement a real-time color-based object segmenter using trackbars.

# The Concept: Why HSV over BGR?
- While computers use BGR (Blue, Green, Red) to display images, it is a poor choice for color detection because a change in lighting significantly changes all three values. 

- HSV (Hue, Saturation, Value) is superior for computer vision because:
- Hue (0-179): Represents the "color" itself (Red, Orange, Yellow, etc.).
- Saturation (0-255): Represents the "vibrancy" or "purity" of the color.
- Value (0-255): Represents the "brightness" or "intensity."

- By isolating the Hue, we can detect colors regardless of shadows or bright highlights.

#Tools Used:
- Python
- OpenCV
- NumPy

# Features
- Live Conversion: Toggle between BGR, Grayscale, HSV, and Lab spaces.
- Interactive Trackbars: 6 sliders to adjust High/Low HSV ranges in real-time.
- Binary Masking: Visualizes the "Mask" (the logical decision-making process of the computer).
- Bitwise Result: Shows the final isolated object after applying the mask to the original frame.