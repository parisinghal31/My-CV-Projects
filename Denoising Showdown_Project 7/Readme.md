# Image Denoising and Restoration Benchmark

- A comprehensive Python framework that injects three distinct variants of mathematical noise into a clean input image and attempts recovery using five classical and modern spatial/non-local filtering techniques. Performance is tracked and quantified using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).



## Tools Used

- This project relies on a robust scientific Python stack engineered for performance, matrix mathematics, and data visualization:

* Python: The foundational programming language used for script execution.

* OpenCV (Open Source Computer Vision Library): Utilized for high-performance image processing operations, including grayscale transformation, spatial filtering (`cv2.blur`, `cv2.GaussianBlur`, `cv2.medianBlur`), and edge-preserving filtering (`cv2.bilateralFilter`).

* NumPy: The core library for scientific computing. Used to handle image representations as multi-dimensional matrices, implement statistical noise distributions, and manage vector arrays.

* Scikit-Image (skimage): A collection of algorithms for image processing. Leveraged for advanced restoration calculations (`denoise_nl_means`, `estimate_sigma`) and accurate image quality assessments (`peak_signal_noise_ratio`, `structural_similarity`).

* Matplotlib: A comprehensive data visualization library used to compile, arrange, and render the final 3 times 7 benchmark evaluation matrix plot.



## Real-World Applications

- Understanding which filter counteracts specific noise profiles is vital across several high-impact technology sectors:

### 1. Medical Imaging (MRI, CT, Ultrasound)
* Context: Medical scans are inherently prone to Rician and Poisson (shot) noise due to physics limitations or minimized radiation dosages. 

* Application: Techniques like Bilateral Filtering and Non-Local Means are critical because blurring a clinical image could erase a tiny tumor or micro-fracture. Preserving razor-sharp structural boundaries while eliminating low-light grain saves lives.


### 2. Deep Space Photography & Astronomy

* Context: Digital sensors on telescopes like Hubble or James Webb capture images across billions of light-years. Because photon counts are incredibly sparse over long distances, Poisson noise heavily degrades the raw feed.

* Application: Advanced patch-matching filters (like NLM) find recurring cosmic structural textures, allowing astronomers to clean the images without inventing false celestial bodies.


### 3. Faulty Hardware & Heritage Document Archiving


* Context: Old photographs scanned from degraded physical film often contain extreme scratches, dirt specs, or localized sensor "dead pixel" dropouts, which behave identically to Salt-and-Pepper noise. 

* Application: Passing these images through a Median Filter smoothly eliminates speckles and dust spots seamlessly, restoring historic texts and portraits without corrupting the overarching media quality.


### 4. Consumer Smartphone Night-Mode Photography

* Context: Small smartphone camera sensors must push ISO levels to the extreme in low-light environments, creating heavy thermal Gaussian noise (color grain).

* Application: Modern mobile computational photography pipelines utilize multi-scale Gaussian and Bilateral filters in real-time to smooth out background noise in shadows while keeping your face and hair sharply in focus.



## Core Analysis Report

### 1. Noise vs Filter Efficacy Matrix

| Filter Type | Gaussian Noise | Salt-and-Pepper Noise | Poisson Noise |
| :--- | :--- | :--- | :--- |
| Mean Filter | Poor (Introduces edge blur) | Terrible (Spreads the noise spike out) | Poor (Blurs variance details) |
| Median Filter | Poor (Patchy Artifacts) | Excellent / Optimal Match | Poor (Damages gradients) |
| Gaussian Filter | Good (Mathematically Consistent) | Bad (Blurs out pixel spikes into blobs) | Moderate (Smooths variance) |
| Bilateral Filter | Excellent (Preserves Edges) | Terrible (Preserves noise spikes) | Excellent (Smooths flat zones) |
| Non-Local Means | Superior (Best Textures/SSIM) | Poor (Confuses noise for textures) | Superior (Excellent Reconstruction) |


### 2. Analytical Takeaways

1.  Salt & Pepper vs The Median Filter:

- Impulse noise corrupts single pixel elements to extreme values. Because these values act as dramatic outliers, sorting neighbor elements pushes them to the far ends of the array, meaning the `Median Filter` drops them entirely. Linear filters like `Mean` or `Gaussian` blend these extreme values into neighboring regions, turning pin-drop points into soft, ruined blur circles.

2.  Structural Preservation (Bilateral vs NLM):

- While a `Gaussian Filter` dampens high-frequency Gaussian noise, it degrades structural boundaries. The `Bilateral Filter` mitigates this by evaluating structural affinity before assigning spatial weights. `Non-Local Means` outperforms them all on structured scenes by checking non-contiguous patterns across a broader window, leading to higher final SSIM metrics.