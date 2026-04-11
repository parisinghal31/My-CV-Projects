# Project 2: Image Arithmetic from Scratch
- Writing my own add, subtract, blend, and bitwise ops using only NumPy. Then comparing with cv2.add vs
  numpy + and explaining saturation vs modulo wraparound with actual output images. This one tiny project
saves you from 100 future bugs.

## Goal
- To understand the difference between **Saturation** (OpenCV) and **Modulo** (NumPy) arithmetic.

## Key Lessons
- `uint8` overflow causes "wraparound" glitches. //every pixel is an unsigned 8 bit int {0-255}
- OpenCV's `cv2.add()` clips values at 255.
- Image blending uses the formula: (ImageA \times \alpha) + (ImageB \times (1 - \alpha))
- NumPy: If a pixel is at 250 (nearly white) and you add 10 to it, the math says 260.
          It "wraps around" back to the start. 260 becomes 4 {260 % 256}.
- OpenCV:  It "saturates." If the result is > 255, it just stays at 255.