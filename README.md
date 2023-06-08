# Metin2 Metin Farmbot

This script is designed to recognize Metin stones in Metin without relying on game hooks. It utilizes a pre-trained model to perform object detection on images and identifies the location of Metin stones within the game world.

## Features

- Detection of Metin stones in Metin without game hooks
- Bounding box visualization of detected Metin stones
- Confidence score for each detected Metin stone

## Requirements

  - Python 3.7 or higher
  - TensorFlow 2.x
  - OpenCV
  - NumPy
  - PIL (Python Imaging Library)

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/metin-stone-recognition.git
   ```
2. Install the required dependencies:
  
   ``` shell
   pip install -r requirements.txt
   ```
   
## Usage
1.   Run the script by executing:

   ```shell
   python main.py path/to/input_image.png
   ```
2. The script will perform Metin stone recognition and display the results in the console.
3. If a Metin stone with a confidence score higher than 0.7 is detected, an image with a bounding box will be saved as image_with_box.png.

## Requirements

The script relies on the following Python packages:
   - tensorflow
   - cv2
   - numpy
   - Pillow

For detailed package versions, refer to the requirements.txt file.