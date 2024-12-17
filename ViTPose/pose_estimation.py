import os
import cv2
import json
import argparse
from tqdm import tqdm
import numpy as np
from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result

def inference_and_render_skeleton(image_paths, bbox_list, output_path, config_file, checkpoint_file):
    """
    Perform pose estimation on multiple inpainted images and render the skeletons.

    Args:
        image_paths (list): List of paths to the inpainted images.
        bbox_list (list): List of bounding boxes corresponding to each image.
        output_path (str): Directory to save the processed images with pose results.
        config_file (str): Path to the config file for the pose model.
        checkpoint_file (str): Path to the checkpoint file for the pose model.
    """
    os.makedirs(output_path, exist_ok=True)

    # Initialize the pose model
    pose_model = init_pose_model(config_file, checkpoint_file, device='cuda:0') 

    line_thickness = 3
    keypoint_radius = 4

    output_data = []

    for image_path, bbox in tqdm(zip(image_paths, bbox_list), desc="Processing images"):
        x_min, y_min, x_max, y_max = bbox
        
        expand_amount = 3
        x_min = max(0, x_min - expand_amount)
        y_min = max(0, y_min - expand_amount)
        x_max = x_max + expand_amount
        y_max = y_max + expand_amount

        width = x_max - x_min
        height = y_max - y_min

        # Add dummy confidence to the bounding box
        person_results = [{'bbox': [x_min, y_min, width, height] + [1]}]

        # Perform inference for pose estimation
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            image_path,
            person_results=person_results,
            format='xywh',
            bbox_thr=0.5,
            dataset='TopDownOCHumanDataset',  # Set for OCHuman
            return_heatmap=False
        )

        # Visualize the pose results on the image
        vis_result = vis_pose_result(
            pose_model,
            image_path,
            pose_results,
            dataset='TopDownOCHumanDataset',
            show=False,
            kpt_score_thr=0.5,  # Confidence threshold for keypoints
            radius=keypoint_radius,
            thickness=line_thickness
        )


        # Save the image with drawn keypoints and skeletons
        output_image_path = os.path.join(output_path, os.path.basename(image_path))
        cv2.imwrite(output_image_path, vis_result)
        print(f"Saved pose visualization: {output_image_path}")

        for result in pose_results:  # Loop over all detected persons (in case of multiple detections)
            result['keypoints'] = result['keypoints'].astype(np.float64)
            keypoints = []

            for kp in result['keypoints']:  # Keypoints is a list of (x, y, visibility)
                x, y, visibility = kp
                keypoints.extend([x, y, visibility])
                    
            # Use dummy values for center and scale, doesn't affect evaluation
            output_entry = {
                "category_id": 1,
                "center": [0, 0],
                "image_id": 1,  # Placeholder, can be adjusted as needed
                "keypoints": keypoints,
                "scale": [0, 0],
                "score": 1.0
            }

            output_data.append(output_entry)

    # Save all pose results to a JSON file
    annotations_path = os.path.join(output_path, 'pose_annotations.json')
    with open(annotations_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Annotations saved to {annotations_path}")


def main():
    parser = argparse.ArgumentParser(description="Process inpainted images to perform pose estimation.")
    parser.add_argument('--image_folder', type=str, help="Folder containing the inpainted images.")
    parser.add_argument('--output_folder', type=str, help="Folder to save the inpainted images with pose results.")
    parser.add_argument('--bbox_list_path', type=str, help="Path to the JSON file containing bboxes list.")
    parser.add_argument('--config_file', type=str, help="Path to the pose model config file.")
    parser.add_argument('--checkpoint_file', type=str, help="Path to the pose model checkpoint file.")
    
    args = parser.parse_args()

    # Collect all image paths from the folder
    image_paths = sorted([os.path.join(args.image_folder, file) for file in os.listdir(args.image_folder)])

    # Read the bounding boxes from the JSON file
    with open(args.bbox_list_path, 'r') as f:
        bbox_list = json.load(f)

    # Perform pose estimation and render skeletons on images
    inference_and_render_skeleton(image_paths, bbox_list, args.output_folder, args.config_file, args.checkpoint_file)

if __name__ == '__main__':
    main()
