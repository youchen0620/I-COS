## Environment
 ``` bash
 pip install -r requirements.txt
 ```

## Inference 

#### 1. OneFormer
   ✨To apply segmentation with Oneformer, just run:
   ``` 
   python segmentation.py -v
   ``` 
   You will then see a comparison of the before and after segmentation results, which are saved in the `./outputs` directory by default. To modify the default settings, you can refer to the help information by running `python segmentation.py --help`.
    Note: We use the checkpoint [shi-labs/oneformer_coco_swin_large](https://huggingface.co/shi-labs/oneformer_coco_swin_large).  
     
#### 2. Salient Human Segmentation  
   Since OneFormer does not have the capability to segment only people in the foreground, nor does it function like SAM2, which can perform segmentation with the aid of hints such as boxes, scribbles, or points, we instead leverage a salient human detection model to achieve this. 
   The model we use for inference is [U^2-Net](https://huggingface.co/shi-labs/oneformer_coco_swin_large). 

   ✨Step 1: To apply salient human detection and obtain the corresponding masks, please follow the instructions provided by [U^2-Net](https://huggingface.co/shi-labs/oneformer_coco_swin_large). An excerpt is provided below:
>   **(2021-Feb-06)** Recently, some people asked the problem of using U<sup>2</sup>-Net for human segmentation, so we trained another example model for human segemntation based on [**Supervisely Person Dataset**](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets). <br/>
(1) To run the human segmentation model, please first downlowd the [**u2net_human_seg.pth**](https://drive.google.com/file/d/1m_Kgs91b21gayc2XLW0ou8yugAIadWVP/view?usp=sharing) model weights into ``` ./saved_models/u2net_human_seg/```. <br/>
(2) Prepare the to-be-segmented images into the corresponding directory, e.g. ```./test_data/test_human_images/```. <br/>
(3) Run the inference by command: ```python u2net_human_seg_test.py``` and the results will be output into the corresponding dirctory, e.g. ```./test_data/u2net_test_human_images_results/```<br/><br/>
  

   So, the generated masks will be placed in directory `./U-2-Net/test_data/u2net_test_human_images_results` by default.  
  
   ✨Step 2: To further utilize the masks generated above, we use them to blur the background, placing greater emphasis on the people in the foreground: 
   ``` 
   python blurBG.py
   ```
   Then, the blurred images will be are saved in the `./blur_images` directory by default. To modify the default settings, you can refer to the help information by running `python blurBG.py --help`.  

   ✨Step 3: To apply segmentation with Oneformer on the foreground-focused images instead of the original images, similar to what was previously mentioned, just run:
   ``` 
   python segmentation.py -v -g ./blur_images -s ./outputs_blur
   ``` 
   You will then see a comparison of the before and after segmentation results, which are saved in the `./outputs_blur` directory. 
   