# Image Manipulation Program

## Table of Contents
1. [Description](#description)
2. [Features](#features)
3. [Usage](#usage)
4. [Dependencies](#dependencies)
5. [Assumptions](#assumptions)
6. [Contributions](#contributions)
7. [License](#license)

## Description
This Image Manipulation Program is a Python-based application that allows you to perform various operations on images. It provides a user-friendly interface for tasks related to image details, resizing, rotation, intensity transforms, image analysis, filtering, image conversion, steganography, encryption, and watermarking. Additionally, it offers the option to save images after applying modifications.
Below is a detailed description of the program's features and functionalities.

## Features
### 1. Image Details
   - Provides information about the loaded image:
     - Width
     - Height
     - Format (e.g., JPEG, PNG)
   - Displays histograms for both grayscale and RGB channels.

### 2. Image Resizing
   - Allows users to resize the image while maintaining aspect ratio.

### 3. Image Rotation
   - Users can rotate the image by 90-degree increments.

### 4. Intensity Transforms
   a. **Histogram Equalization** (Implemented)
      - Enhances the contrast of the image.
   b. **Histogram Matching**
      - Allows the user to match the image histogram to a reference histogram.
   c. **Gamma Correction**
      - Adjusts the gamma value to control brightness and contrast.
   d. **Log Transformation**
      - Enhances details in the darker regions of the image.

### 5. Image Analysis
   - Displays edge maps using various highpass filters for edge detection.

### 6. Filtering
   a. **Smoothing**
      - Provides options for box and Gaussian smoothing filters.
   b. **Sharpening**
      - Enhances image details and edges.
   c. **Color Mapping**
      i. **Cold Filter**
         - Applies a cool color filter to the image.
      ii. **Warm Filter**
         - Applies a warm color filter to the image.

### 7. Image Conversion
   - Allows users to convert between different color spaces (e.g., RGB, grayscale, HSV).

### 8. Steganography
   - **Encode Function**
      - Embeds hidden data (text, image, etc.) into the image.
   - **Decode Function**
      - Extracts hidden data from the image based on a predefined assumption.

### 9. Encryption
   - **Encrypt Function**
      - Encrypts the image using a chosen encryption algorithm and key.
   - **Decrypt Function**
      - Decrypts the encrypted image using the same encryption key.

### 10. Watermarking
   - Users can add text or logos as watermarks to the image.

### 11. Save Images
   - Provides the option to save the modified images after performing various operations.

## Usage
To use the program, follow the instructions in the user interface to load an image and select the desired operation from the menu. Save the modified image when necessary.

## Dependencies
- Python 3.x
- OpenCV (for image manipulation)
- NumPy (for numerical operations)
- Matplotlib (for plotting histograms)
- Other libraries based on the specific functionalities implemented (e.g., cryptography library for encryption)

## Assumptions
- For steganography, assume a basic LSB (Least Significant Bit) algorithm for encoding and decoding.
- For encryption, assume a symmetric encryption algorithm like AES (Advanced Encryption Standard).

## Contributions
Contributions and improvements to this image manipulation program are welcome. Feel free to fork the repository, make changes, and submit pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
