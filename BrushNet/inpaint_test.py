from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
import torch
import cv2
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import json

base_model_path = "data/ckpt/realisticVisionV60B1_v51VAE"
brushnet_path = "data/ckpt/random_mask_brushnet_ckpt"

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

with open("../dataset/image_ids.json", "r", encoding="utf-8") as fi:
    image_ids = json.loads(fi.read())
    
    for image_id in tqdm(image_ids[2500:]):
        image_paths = glob.glob("../dataset/images/" + image_id + ".jpg")
        mask_paths = []
        for image_path in image_paths:
            mask_pattern = f"../dataset/masks/mask*_{image_path.split('/')[-1]}"
            found_masks = glob.glob(mask_pattern)
            mask_paths.append(found_masks)

        for image_path, mask_sublist in zip(image_paths, mask_paths):
            if len(mask_sublist) == 0:
                continue

            for mask_path in mask_sublist:
                caption = "image repair with landscape"
                negative_prompt = "human"
                brushnet_conditioning_scale = 1.0

                init_image = cv2.imread(image_path)[:, :, ::-1]
                input_mask = cv2.imread(mask_path)
                mask_image = 1.*(input_mask.sum(-1)<255)[:,:,np.newaxis]
                init_image = init_image * (1 - mask_image)

                init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
                mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")

                generator = torch.Generator("cuda").manual_seed(1234)

                image = pipe(
                    caption,
                    init_image,
                    mask_image,
                    num_inference_steps=50,
                    generator=generator,
                    brushnet_conditioning_scale=brushnet_conditioning_scale,
                    negative_prompt=negative_prompt
                ).images[0]

                cv2.imwrite("../dataset/test/inpainted" + mask_path.split('/')[-1].split("_")[0].replace("mask", "") + "_" + image_id + ".jpg", np.array(image))
