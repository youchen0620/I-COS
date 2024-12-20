import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_pose(pred_file, gt_file):
    """
    Evaluate the pose estimation results using COCO metrics.

    Parameters:
        pred_file (str): Path to the JSON file containing predicted keypoints.
        gt_file (str): Path to the JSON file containing ground truth keypoints.

    Returns:
        None
    """
    coco_gt = COCO(gt_file)
    coco_pred = coco_gt.loadRes(pred_file)

    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt, coco_pred, iouType='keypoints')

    # Evaluate
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Print mAP (Mean Average Precision)
    print("mAP:", coco_eval.stats[0])  # AP (Average Precision) @ IoU=0.50:0.95

def main():
    parser = argparse.ArgumentParser(description="Evaluate pose estimation results using COCO metrics")
    parser.add_argument('--pred_file', type=str, required=True, help="Path to the predicted keypoints JSON file")
    parser.add_argument('--gt_file', type=str, required=True, help="Path to the ground truth keypoints JSON file")
    
    args = parser.parse_args()

    evaluate_pose(args.pred_file, args.gt_file)

if __name__ == "__main__":
    main()
