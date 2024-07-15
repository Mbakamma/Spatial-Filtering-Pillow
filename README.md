Image Processing with Pillow: Geometric Operations and Mathematical Tools
Overview
This project demonstrates the application of various geometric operations and mathematical tools for image processing using the Python Pillow library. The focus is on spatial operations, which involve using neighboring pixel values to determine the value of the current pixel. These operations are fundamental in many computer vision tasks, such as filtering, sharpening, edge detection, and segmentation, and are crucial for building Artificial Intelligence algorithms.

Key Operations
Loading and Displaying Images:

Images are loaded and displayed using Pillow and Matplotlib to visualize changes during processing.
Adding Noise to Images:

Gaussian noise is added to images to simulate real-world imperfections, aiding in the demonstration of noise reduction techniques.
Filtering Noise:

Mean Filtering: A smoothing filter that averages the pixel values in a neighborhood to reduce noise.
Gaussian Blur: A filter that applies a Gaussian function to blur the image, effectively reducing noise while maintaining image integrity.
Image Sharpening:

Custom kernels and predefined filters are used to enhance the edges of objects within an image, making the image appear clearer.
Edge Detection:

Techniques like edge enhancement and edge finding are used to highlight the boundaries of objects within an image, aiding in object detection and image segmentation.
Median Filtering:

This filter replaces each pixel value with the median value of the pixels in a neighborhood, which is particularly effective for removing 'salt and pepper' noise from images.
Tools and Libraries Used
Pillow: For image processing operations such as filtering, blurring, and sharpening.
Matplotlib: For visualizing the original and processed images.
NumPy: For creating and manipulating kernels for filtering operations.

Sample Code
Here's a brief snippet demonstrating the application of Gaussian Blur to reduce noise in an image:
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image = Image.open("lenna.png")

# Add Gaussian noise
rows, cols = image.size
noise = np.random.normal(0, 15, (rows, cols, 3)).astype(np.uint8)
noisy_image = np.array(image) + noise
noisy_image = np.mod(noisy_image, 256)  # Ensure values are within [0, 255]
noisy_image = Image.fromarray(noisy_image)

# Apply Gaussian Blur
image_filtered = noisy_image.filter(ImageFilter.GaussianBlur(4))

# Plot the images
def plot_image(image_1, image_2, title_1="Original", title_2="New Image"):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1)
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2)
    plt.title(title_2)
    plt.show()

plot_image(image, image_filtered, title_1="Noisy Image", title_2="Filtered Image")


This project provides a comprehensive overview of essential image processing techniques using Pillow. These operations form the basis for more advanced computer vision tasks and are vital for enhancing image quality, reducing noise, and detecting features within images. By understanding and applying these techniques, developers can build more robust and efficient image processing applications.
