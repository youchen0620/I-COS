import numpy as np
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import torch
from tqdm import tqdm


processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")
model.eval()


def process_single_image(filename, human_labelId, args):
    '''
    Process one image to get masks
    '''
    ## Segmentation
    image = Image.open(os.path.join(args.seg_dir, filename))
    inputs = processor(image, ["panoptic"], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs) 
    predicted_panoptic = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]], label_ids_to_fuse = set())
    predicted_panoptic_map = predicted_panoptic[0]["segmentation"]
    
    ## Visualization
    if args.visualize:
        show_image_comparison(image, predicted_panoptic_map, os.path.join(args.save_dir, filename))

    ## Masking
    labelIds = [segment['label_id'] for segment in predicted_panoptic[0]['segments_info']]
    # extract id == (label_id == 'human')
    idIsHuman = [(i+1) for i, labelId in enumerate(labelIds) if labelId == human_labelId]

    # create mask_dir
    if args.masking:
        mask_dir = os.path.join(args.save_dir, filename.split('.')[0])
        os.makedirs(mask_dir, exist_ok=True)
        
        for i, _ in enumerate(idIsHuman):
            idIsHuman_subset = [id for j, id in enumerate(idIsHuman) if j != i]
            bool_mask = ~(torch.isin(predicted_panoptic_map, torch.tensor(idIsHuman_subset)))
            int_mask = bool_mask.byte() * 255
            image_mask = Image.fromarray(int_mask.numpy())
            image_mask.save(f'{mask_dir}/{filename.split(".")[0]}_{i+1}.jpg')


def show_image_comparison(image, predicted_map, save_path):
    '''
    Displays the original image and the segmented image side-by-side.

    Args:
        image (PIL.Image): The original image.
        predicted_map (PIL.Image): The segmented image.
        segmentation_title (str): The title for the segmented image.
    '''

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_map)
    plt.title("Oneformer Segmentation")
    plt.axis("off")
    plt.savefig(save_path)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--masking", action="store_true", help='whether to get masking (mask other people in the image)')
    parser.add_argument("-v", "--visualize", action="store_true", help='visualization of predicted_map')
    parser.add_argument("-d", "--data_dir", default='./images', type=str, help="path of directory of OCHuman image dataset")
    parser.add_argument("-g", "--seg_dir", default='./images', type=str, help="path of directory of images you want to segment (by defualt, same as --data_dir)")
    parser.add_argument("-s", "--save_dir", default='./outputs', type=str, help="path of directory for saving predicted_map")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    ## Load one reference image to extract human label ID
    image = Image.open(os.path.join(args.data_dir, '000008.jpg'))
    inputs = processor(image, ["panoptic"], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_panoptic = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image.size[::-1]], label_ids_to_fuse=set()
    )
    predicted_panoptic_map = predicted_panoptic[0]["segmentation"]

    # extract (label_id == 'human')
    labelIds = [segment['label_id'] for segment in predicted_panoptic[0]['segments_info']]
    values, counts = np.unique(labelIds, return_counts=True)
    human_labelId = values[list(counts).index(5)]

    ## Iterate
    for filename in tqdm(os.listdir(args.seg_dir)):
        process_single_image(filename, human_labelId, args)
        