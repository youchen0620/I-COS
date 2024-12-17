import cv2
import json
import argparse
import numpy as np

SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], 
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], 
    [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], 
    [4, 6], [5, 7]
]
# Adjust the skeleton indices to zero-based indexing
for i in range(len(SKELETON)):
    SKELETON[i][0] -= 1
    SKELETON[i][1] -= 1

def draw_skeleton(image, keypoints, thickness=2, radius=5):
    """
    Draw keypoints and skeleton on the image.

    Args:
        image (numpy.ndarray): The image on which skeleton will be drawn.
        keypoints (list): A list of keypoints (x, y, visibility).
        thickness (int): The thickness of the lines to be drawn for skeleton.
        radius (int): The radius of the keypoints to be drawn.
    """
    # Loop through and draw keypoints
    for i in range(0, len(keypoints), 3):
        x, y, visibility = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        if visibility > 0.5:  # Only draw keypoints with visibility > 0.5
            cv2.circle(image, (int(x), int(y)), radius, (0, 255, 0), -1)

    # Loop through and draw skeleton connections based on SKELETON structure
    for joint_start, joint_end in SKELETON:
        x_start, y_start, visibility_start = keypoints[joint_start * 3], keypoints[joint_start * 3 + 1], keypoints[joint_start * 3 + 2]
        x_end, y_end, visibility_end = keypoints[joint_end * 3], keypoints[joint_end * 3 + 1], keypoints[joint_end * 3 + 2]

        if visibility_start > 0.5 and visibility_end > 0.5:
            cv2.line(image, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (0, 255, 0), thickness)

def process_image(image_path, annotations):
    """
    Process an image by drawing skeletons based on annotations.

    Args:
        image_path (str): The path to the original image.
        annotations (list): A list of annotations for the skeletons.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Loop through annotations and draw skeletons
    for annotation in annotations:
        keypoints = annotation['keypoints']
        draw_skeleton(image, keypoints)

    return image

def main():
    # Set up argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Draw skeletons on images based on annotations.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the original image.")
    parser.add_argument('--annotations_file', type=str, required=True, help="Path to the JSON file containing annotations.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the image with skeletons drawn.")
    args = parser.parse_args()

    with open(args.annotations_file, 'r') as f:
        annotations = json.load(f)

    result_image = process_image(args.image_path, annotations)

    cv2.imwrite(args.output_path, result_image)
    print(f"Skeleton image saved to {args.output_path}")

if __name__ == '__main__':
    main()
