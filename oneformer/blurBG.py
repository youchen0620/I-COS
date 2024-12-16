from PIL import Image, ImageFilter
import argparse
import os
from tqdm import tqdm


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default='./images', type=str, help="path of directory of OCHuman image dataset")
    parser.add_argument("-u", "--u2net_mask_dir", default='./U-2-Net/test_data/u2net_test_human_images_results', type=str, help="path of directory for foregound-masks produced by u2net")
    parser.add_argument("-s", "--save_dir", default='./blur_images', type=str, help="path of directory for saving blurred images")
    parser.add_argument("-b", "--blur_intensity", default=80, type=int, help="radius for blur intensity")
    parser.add_argument("-t", "--threshlod", default=200, type=int, help="the threshold for binarize mask")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    for filename in tqdm(os.listdir(args.u2net_mask_dir)):

        # Load the image and mask
        image = Image.open(os.path.join(args.data_dir, filename.split('.')[0]+'.jpg'))
        mask = Image.open(os.path.join(args.u2net_mask_dir, filename)).convert("L")  # Convert mask to grayscale
        binarized_mask = mask.point(lambda p: 255 if p > args.threshlod else 0) # Binarize mask

        # Create a blurred version of the image
        blurred_image = image.filter(ImageFilter.GaussianBlur(args.blur_intensity)) 

        # Composite the images using the mask
        result = Image.composite(image, blurred_image, binarized_mask)

        # Save or display the result
        result.save(os.path.join(args.save_dir, filename.split('.')[0]+'.jpg'))
