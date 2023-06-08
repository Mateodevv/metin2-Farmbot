import argparse
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image


def draw_bounding_box(image_in, ymin, xmin, ymax, xmax, color=(0, 255, 0), thickness=2):
    # Convert the image to numpy array
    image_np = np.array(image_in)

    # Convert ymin, xmin, ymax, xmax to pixel values
    height, width, _ = image_np.shape
    ymin_pix = int(ymin * height)
    xmin_pix = int(xmin * width)
    ymax_pix = int(ymax * height)
    xmax_pix = int(xmax * width)

    # Draw the bounding box on the image
    cv2.rectangle(image_np, (xmin_pix, ymin_pix), (xmax_pix, ymax_pix), color, thickness)

    # Convert the image back to PIL Image
    image_with_box = Image.fromarray(image_np)

    return image_with_box


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Metin Stone Recognition')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    # Load the SavedModel
    model = tf.saved_model.load('saved_model')

    # Define the input and output tensor names
    output_tensor_name = 'detection_boxes'

    # Load and preprocess the image
    image = tf.io.read_file(args.image_path)
    image = tf.image.decode_png(image, channels=3)  # Adjust channels if necessary
    image = tf.cast(image, tf.uint8)  # Convert to uint8
    image = tf.image.encode_jpeg(image)

    # Create input signatures

    # Run inference
    prediction = model.signatures['serving_default'](image_bytes=tf.constant([image.numpy()]), key=tf.constant([b'']))

    # Access the output tensor
    output_tensor = prediction[output_tensor_name]

    # Load the input image
    image = Image.open(args.image_path)

    # Format and print the predictions in table format
    print("# | Confidence |   y-min  |   x-min  |   y-max  |   x-max  ")
    print("-----------------------------------------------------")
    for i, box in enumerate(output_tensor[0]):
        confidence = prediction['detection_scores'][0][i]
        ymin_rel, xmin_rel, ymax_rel, xmax_rel = box.numpy()
        print(f"{i + 1} |   {confidence:.4f}   |  {ymin_rel}  |  {xmin_rel}  |  {ymax_rel}  |  {xmax_rel}  ")
        if confidence > 0.7:
            img = draw_bounding_box(image, ymin_rel, xmin_rel, ymax_rel, xmax_rel)
            output_path = 'image_with_box.png'
            img.save(output_path)
            print(f"Image with bounding box saved to: {output_path}")


if __name__ == '__main__':
    main()
