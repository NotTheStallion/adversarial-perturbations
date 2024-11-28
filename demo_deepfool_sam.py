import torch
from segment_anything import sam_model_registry
import cv2
from segment_anything import SamAutomaticMaskGenerator
import supervision as sv
import matplotlib.pyplot as plt
import numpy as np

def show_anns(anns):
    if len(anns) == 0:
        return
    # Sort annotations by area in descending order
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    print(sorted_anns[2]['segmentation'])
    sorted_anns = sorted_anns[2:3]
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Create an RGBA image with ones and set alpha channel to 0
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        # Generate a random color mask with alpha channel set to 0.35
        color_mask = np.concatenate([np.random.random(3), [0.9]])
        img[m] = color_mask
    # Display the image with masks
    ax.imshow(img)

IMAGE_PATH = "data/demo_deepfool/test_img6.jpg"  # Replace with your image file
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"  # Replace with the path to the SAM checkpoint file


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam, output_mode='binary_mask')

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
result = mask_generator.generate(image_rgb)


# print(result)
plt.figure(figsize=(20,20))
plt.imshow(image_rgb)
show_anns(result)
plt.axis('off')
plt.show() 


sorted_anns = sorted(result, key=(lambda x: x['area']), reverse=True)
print(sorted_anns[2]['segmentation'])



    