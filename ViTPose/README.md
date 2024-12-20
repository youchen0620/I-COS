## Setup
To set up the environment, run:
```bash
pip install -r requirements.txt
```

## Inference 

### 1. Bounding Box Calculation

The `bbox_from_masks.py` script generates bounding boxes from masks and overlays them onto images.

#### Example usage: 
```
python bbox_from_masks.py \
  --mask_dir ./masks/ \
  --image_dir ./inpainted_images/ \
  --output_json ./bbox.json \
  --output_mask_dir ./masks_w_boxes/ \
  --output_image_dir ./inpainted_images_w_boxes/
```

#### Arguments
- --mask_dir: Directory containing the masks.
- --image_dir: Directory containing the inpainted images.
- --output_json: Path to save the bounding boxes list in JSON format.
- --output_mask_dir: Directory to save the masks with bounding boxes drawn on them.
- --output_image_dir: Directory to save the inpainted images with bounding boxes drawn on them.

#### Notes
- Ensure that the image names in `./masks/` and `./inpainted_images/` correspond to each other. The images will be sorted by their file names, and the bounding boxes will be applied to the matching images accordingly.


### 2. Pose Estimation

The `pose_estimation.py` script performs pose estimation on input images, using a pre-trained model to predict human poses and render the results.

#### Example Usage
```bash
python pose_estimation.py \
  --image_folder ./inpainted_images/ \
  --output_folder ./rendered_images/ \
  --bbox_list_path ./bbox.json \
  --config_file ../mmpose_configs/ViTPose_base_ochuman_256x192.py \
  --checkpoint_file vitpose_base_coco_aic_mpii.pth
```

#### Arguments

- --image_folder: Directory containing the input inpainted images.
- --output_folder: Directory to save the rendered images with pose estimates.
- --bbox_list_path: Path to the JSON file containing bounding boxes used for pose estimation.
- --config_file: Path to the configuration file for ViTPose (e.g., ViTPose_base_ochuman_256x192.py).
- --checkpoint_file: Path to the checkpoint file for the pre-trained ViTPose model.

#### Notes
- We use the configuration file [ViTPose_base_coco_256x192.py](https://github.com/ViTAE-Transformer/ViTPose/blob/main/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py) from the ViTPose repository.
- The checkpoint file used is [ViTPose-B](https://onedrive.live.com/?authkey=%21AMGk2fMQhRTN0o4&id=E534267B85818129%2125500&cid=E534267B85818129&parId=root&parQt=sharedby&o=OneUp).


#### Alternative: pose_estimation_baseline.py
The `pose_estimation_baseline.py` script operates similarly to `pose_estimation.py`, but it leverages the mmpose API for inference on a single image. In contrast, our pipeline approach performs inference across multiple inpainted images. To adapt the baseline script for our use case, simply replace the `--image_folder` argument with `--image_path`, and specify the path to the original image instead of the inpainted images.

### 3. Render Skeleton

The `draw_skeleton.py` script renders the predicted skeletons on images based on pose annotations.

#### Example Usage
```bash
python draw_skeleton.py \
  --image_path ./original.jpg \
  --annotations_file ./rendered_images/pose_annotations.json \
  --output_path ./rendered_images/predictions.jpg
```

#### Arguments
- --image_path: Path to the input image on which the skeleton will be drawn.
- --annotations_file: Path to the JSON file containing pose annotations (e.g., pose_annotations.json).
- --output_path: Path where the image with the rendered skeleton will be saved.

## Evaluation

The evaluation script computes pose estimation performance using COCO metrics, specifically focusing on Average Precision (mAP) for keypoint detection.

### Usage:
To evaluate the pose estimation results, run the following command:

```bash
python evaluate_pose_estimation.py\
--pred_file /path/to/predictions.json\
--gt_file /path/to/ground_truth.json
```