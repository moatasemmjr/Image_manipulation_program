# Image_manipulation_program

This Image Manipulation Program is a Python-based application that allows you to perform various operations on images. It provides a user-friendly interface for tasks related to image details, resizing, rotation, intensity transforms, image analysis, filtering, image conversion, steganography, encryption, and watermarking. Additionally, it offers the option to save images after applying modifications.

Table of Contents
Features
Requirements
How to Use
Functionality
Steganography
Encryption
Watermarking
Saving Images
Features<a name="features"></a>
This program provides the following features:

Image Details:

Display image width, height, format.
Generate histograms for both gray and RGB channels.
Image Resizing:

Resize images to specified dimensions.
Image Rotation:

Rotate images in 90-degree increments.
Intensity Transforms:

Histogram Equalization.
Histogram Matching.
Gamma Correction.
Log Transformation.
Image Analysis:

Display edge maps using different highpass filters.
Filtering:

Smoothing (box and Gaussian).
Sharpening.
Color Mapping (Cold and Warm filters).
Image Conversion:

Convert between different color spaces.
Steganography:

Encode and decode hidden messages within images.
Encryption:

Encrypt and decrypt images to protect their content.
Watermarking:

Add text or logos as watermarks to images.
Save Images:

Save modified images to disk.
Requirements<a name="requirements"></a>
To run this program, you will need:

Python 3.x
The following Python libraries: OpenCV, NumPy, SciPy, PIL (Pillow)
Additional libraries for encryption and steganography (if used)
You can install the required Python libraries using pip:

bash
Copy code
pip install opencv-python numpy scipy pillow
How to Use<a name="how-to-use"></a>
Clone this repository to your local machine.
Install the required libraries as mentioned in the requirements section.
Run the main.py file to launch the program.
Use the user-friendly interface to select images and apply various image manipulation operations.
Functionality<a name="functionality"></a>
Here is a brief overview of how to use each functionality:

Image Details:

Load an image to display its width, height, and format.
Generate histograms for both gray and RGB channels.
Image Resizing:

Choose the dimensions (width and height) for resizing.
Click the "Resize" button to apply the changes.
Image Rotation:

Click the "Rotate" button to rotate the image by 90 degrees.
Intensity Transforms:

Select the desired transform (Histogram Equalization, Histogram Matching, Gamma Correction, Log Transformation).
Adjust parameters if needed.
Click the "Apply" button to transform the image.
Image Analysis:

Choose the highpass filter type (e.g., Sobel, Laplacian).
Click the "Apply" button to generate edge maps.
Filtering:

Select the filter type (Smoothing or Sharpening).
Choose the specific filter (e.g., Box, Gaussian).
Apply the filter to the image.
Image Conversion:

Convert between different color spaces (e.g., RGB to Grayscale, RGB to HSV).
Steganography:

Encode a message into an image using a secret key.
Decode a hidden message from an encoded image using the same key.
Encryption:

Encrypt an image using a password.
Decrypt the encrypted image with the same password.
Watermarking:

Add text or logos as watermarks to the image.
Adjust transparency and position as needed.
Steganography<a name="steganography"></a>
The steganography functionality allows you to hide and retrieve messages within images. This implementation assumes the use of a secret key for encoding and decoding. Messages are embedded using a steganographic algorithm, and the key is required to retrieve the message.

Encryption<a name="encryption"></a>
The encryption functionality allows you to protect the content of images by encrypting them with a password. The same password is required to decrypt the image and view its content. This provides a level of security for sensitive image data.

Watermarking<a name="watermarking"></a>
You can use the watermarking feature to add text or logos as watermarks to your images. Adjust the transparency and position of the watermark to suit your needs.

Saving Images<a name="saving-images"></a>
After applying any modifications to the images, you have the option to save them to your local disk. Make sure to choose an appropriate file format and provide a meaningful name for the saved image.

Enjoy using the Image Manipulation Program! Feel free to contribute to its development and expand its capabilities as needed.
