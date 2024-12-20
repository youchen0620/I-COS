import cv2, os
import numpy as np
import torch

from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import hydra
from hydra import initialize

from tqdm import tqdm
import json

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

np.random.seed(3)

yolo_model = YOLO("yolo11x.pt")

hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize(config_path="./")
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
sam2_predictor = SAM2ImagePredictor(sam2_model)

ImgDir = '../dataset/images/'
with open("../dataset/image_ids.json", "r", encoding="utf-8") as fi:
    image_ids = json.loads(fi.read())

    for image_id in tqdm(image_ids):
        image = cv2.imread(os.path.join(ImgDir, str(image_id).zfill(6) + ".jpg"))
        height, width, channels = image.shape

        results = yolo_model(image)

        sam2_predictor.set_image(image)
        annotations = []
        for result in results:
            confidences = result.boxes.conf.float().tolist()
            class_ids = result.boxes.cls.int().tolist()
            if len(class_ids):
                boxes = result.boxes.xyxy
                for i in range(len(class_ids)):
                    if class_ids[i] == 0 and confidences[i] > 0.3:
                        input_box = np.array(boxes[i].cpu())

                        masks, scores, _ = sam2_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box[None, :],
                            multimask_output=False,
                        )

                        annotations.append(masks[0])
        
        masks = [np.ones((height, width, 3), dtype=np.uint8) * 255 for _ in range(len(annotations))]

        for i, mask in enumerate(annotations):
            canvas = np.zeros(image.shape, image.dtype) + (0, 0, 0)
            masks[i][mask > 0] = canvas[mask > 0]
            masks[i] = 255 - masks[i]

        for i in range(len(masks)):
            cv2.imwrite("../dataset/masks/mask" + str(i) + "_" + str(image_id).zfill(6) + ".jpg", masks[i])