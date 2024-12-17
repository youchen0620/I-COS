import argparse
import numpy as np
import cv2
import os
import json


def mask_to_bbox(mask):
    """
    Compute the bounding box for a single binary mask.

    Args:
        mask (numpy.ndarray): Binary mask where 255 indicates the object.

    Returns:
        list: Bounding box coordinates as a list [x_min, y_min, x_max, y_max].
    """
    rows = np.any(mask == 255, axis=1)
    cols = np.any(mask == 255, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return x_min, y_min, x_max, y_max


def process_single_image_masks(mask_dir, output_json_path):
    """
    Process all masks for a single image to calculate bounding boxes.

    Args:
        mask_dir (str): Path to the directory containing masks for one image.
        output_json_path (str): Path to save bounding boxes as a JSON file.

    Returns:
        list: List of bounding boxes for the given image.
    """
    mask_paths = sorted([os.path.join(mask_dir, file) for file in os.listdir(mask_dir)])
    if not mask_paths:
        raise ValueError(f"No mask files found in directory: {mask_dir}")

    ref_img = cv2.imread(mask_paths[0])
    height, width, _ = ref_img.shape
    mixed_mask = np.zeros((height, width, 3), dtype=np.uint8)

    bg_masks = []
    for mask_path in mask_paths:
        input_mask = cv2.imread(mask_path)
        bg_masks.append(255 - input_mask)
        mixed_mask = cv2.add(mixed_mask, 255 - input_mask)

    mixed_mask = 255 - mixed_mask

    bboxes = []
    for bg_mask in bg_masks:
        bg_mask = 255 - (mixed_mask + bg_mask)
        bbox = mask_to_bbox(bg_mask)
        if bbox is None:
            bbox = [0, 0, width, height]  # Default to whole image if no bbox
        bboxes.append(bbox)

    bboxes = [list(map(lambda x: x.item() if isinstance(x, np.generic) else x, sublist)) for sublist in bboxes]
    with open(output_json_path, 'w') as f:
        json.dump(bboxes, f, indent=4)

    print(f"Bounding boxes saved to {output_json_path}")
    return bboxes

def draw_bboxes_on_images(image_dir, bboxes, output_dir):
    """
    Draw bounding boxes on the images and save the result.

    Args:
        image_dir (str): Path to the directory containing images.
        bboxes (list): List of bounding boxes to draw.
        output_dir (str): Directory to save the images with bounding boxes.
    """
    # Get all image paths from the directory
    image_paths = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir)])
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, image_path in enumerate(image_paths):
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read image at {image_path}. Skipping...")
            continue
        
        # Get the bounding box for this image
        bbox = bboxes[i]
        x_min, y_min, x_max, y_max = bbox

        # Draw the bounding box on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Save the output image with the bounding box
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_image_path, image)
        print(f"Saved image with bounding box to {output_image_path}")




def main():
    parser = argparse.ArgumentParser(description="Process masks to find bounding boxes and draw them on the masks.")
    parser.add_argument('--mask_dir', type=str, help="Directory containing the masks for the image.")
    parser.add_argument('--image_dir', type=str, help="Directory containing the inpainted images for the image.")
    parser.add_argument('--output_json', type=str, help="Path to save the bounding boxes as a JSON file.")
    parser.add_argument('--output_mask_dir', type=str, help="Directory to save the masks with bounding boxes.")
    parser.add_argument('--output_image_dir', type=str, help="Directory to save the inpainted images with bounding boxes.")
    
    args = parser.parse_args()

    # Process the masks and calculate bounding boxes
    bboxes = process_single_image_masks(args.mask_dir, args.output_json)

    # Draw bounding boxes on the masks and inpainted images and save them
    draw_bboxes_on_images(args.mask_dir, bboxes, args.output_mask_dir)
    draw_bboxes_on_images(args.image_dir, bboxes, args.output_image_dir)


if __name__ == '__main__':
    main()
