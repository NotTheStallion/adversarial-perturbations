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
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    # print(sorted_anns[2]['segmentation'])
    sorted_anns = sorted_anns[2:3]
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Create an RGBA image with ones and set alpha channel to 0
    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        # Generate a random color mask with alpha channel set to 0.35
        color_mask = np.concatenate([np.random.random(3), [0.9]])
        img[m] = color_mask
    # Display the image with masks
    ax.imshow(img)


def make_examples():
    # Load pretrained ResNet-34 model
    net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    # Switch to evaluation mode
    net.eval()

    original_images = []
    original_labels = []
    perturbed_images = []
    perturbed_labels = []
    max_pixel_values = []
    diff_norms = []
    region_masks = []

    for i in range(6):
        im_orig = Image.open(f"data/demo_deepfool/test_img6.jpg")

        # Mean and std used for normalization (ImageNet stats)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        original_images.append(
            transforms.Compose(
                [
                    transforms.Resize(256),  # Updated from transforms.Scale
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )(im_orig)
        )

        # Preprocessing the image: resize, crop, convert to tensor, and normalize
        im = transforms.Compose(
            [
                transforms.Resize(256),  # Updated from transforms.Scale
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )(im_orig).cpu()  # Ensure the tensor is on the CPU

        CHECKPOINT_PATH = (
            "sam_vit_h_4b8939.pth"  # Replace with the path to the SAM checkpoint file
        )

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        MODEL_TYPE = "vit_h"

        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam.to(device=DEVICE)

        mask_generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")

        # convert orginal_images[0] to numpy
        image_rgb = cv2.cvtColor(
            np.array(original_images[0].permute(1, 2, 0)), cv2.COLOR_BGR2RGB
        )
        result = mask_generator.generate(image_rgb)

        # print(result)
        # plt.figure(figsize=(20,20))
        # plt.imshow(image_rgb)
        # show_anns(result)
        # plt.axis('off')
        # plt.show()

        sorted_anns = sorted(result, key=(lambda x: x["area"]), reverse=True)
        mask = sorted_anns[i]["segmentation"]
        region_mask = mask.astype(np.uint8)
        region_masks.append(region_mask)

        # plt.imshow((region_mask))
        # plt.imshow(region_mask, cmap='gray')
        # plt.show()

        r, loop_i, label_orig, label_pert, pert_image = deepfool_specific(
            im, net, 413, region_mask=region_mask, num_classes=10, max_iter=1000
        )

        # r, loop_i, label_orig, label_pert, pert_image = local_deepfool(
        #     im, net, num_classes=10, max_iter=1000, region_mask=region_mask
        # )

        # Load class labels from file
        labels = (
            open(os.path.join("data/demo_deepfool/synset_words.txt"), "r")
            .read()
            .split("\n")
        )

        # Get original and perturbed class labels
        str_label_orig = labels[int(label_orig)].split(",")[0]  # Changed np.int to int
        str_label_pert = labels[int(label_pert)].split(",")[0]

        original_labels.append(str_label_orig.split()[1:])
        perturbed_labels.append(str_label_pert.split()[1:])

        # Function to clip tensor values between minv and maxv
        def clip_tensor(A, minv, maxv):
            A = torch.clamp(A, minv, maxv)  # Use torch.clamp for cleaner implementation
            return A

        # Clipping function for images (0-255 range)
        clip = lambda x: clip_tensor(x, 0, 255)

        # Inverse transformation to convert perturbed image back to PIL format
        tf = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0, 0, 0], std=[1 / s for s in std]
                ),  # Reverse normalization
                transforms.Normalize(
                    mean=[-m for m in mean], std=[1, 1, 1]
                ),  # Subtract mean
                transforms.Lambda(clip),  # Clip the values to ensure valid image range
                transforms.ToPILImage(),  # Convert tensor back to PIL image
                transforms.CenterCrop(224),  # Center crop to 224x224
                transforms.ToTensor(),  # Convert back to tensor
            ]
        )

        pert_image = pert_image.view(pert_image.size()[-3:]).type(torch.FloatTensor)
        perturbed_images.append(tf(pert_image))
    return (
        original_images,
        original_labels,
        perturbed_images,
        perturbed_labels,
        max_pixel_values,
        diff_norms,
        region_masks,
    )


def diff(original_images, perturbed_images):
    # Calculate and display the difference between original and perturbed images
    difference_images = []

    for orig, pert in zip(original_images, perturbed_images):
        # Convert images to tensors
        if isinstance(orig, Image.Image):
            orig_tensor = transforms.ToTensor()(orig)
        else:
            orig_tensor = orig
        if isinstance(pert, Image.Image):
            pert_tensor = transforms.ToTensor()(pert)
        else:
            pert_tensor = pert

        # Calculate the difference
        diff_tensor = torch.abs(orig_tensor - pert_tensor)

        # Convert the difference tensor back to a PIL image
        diff_image = transforms.ToPILImage()(diff_tensor)
        difference_images.append(diff_image)

    return difference_images


def plot_diff(original_images, perturbed_images, region_masks=None):
    difference_images = diff(original_images, perturbed_images)

    # Display the difference images with colorbars
    fig_diff, ax_diff = plt.subplots(2, 6, figsize=(20, 5))
    for col in range(6):
        ax_diff[0][col].imshow(transforms.ToPILImage()(region_masks[col]), cmap="gray")
        ax_diff[0][col].set_title(f"Region {col+1}")
        ax_diff[0][col].axis("off")

    for col in range(6):
        # Convert the difference tensor to grayscale for visualization
        diff_im = transforms.ToTensor()(difference_images[col])
        diff_gray = torch.mean(
            diff_im, dim=0
        )  # Convert to grayscale by averaging channels
        im = ax_diff[1][col].imshow(diff_gray, cmap="gray")
        ax_diff[1][col].set_title(f"Difference {col+1}")
        ax_diff[1][col].axis("off")
        fig_diff.colorbar(im, ax=ax_diff[1][col], orientation="vertical")

    fig_diff.suptitle("Difference between Original and Perturbed Images")
    plt.tight_layout()
    plt.show()


def plot_comparaison(
    original_images, perturbed_images, original_labels, perturbed_labels
):
    fig, ax = plt.subplots(2, 6, figsize=(12, 8))
    for col in range(6):
        ax[0][col].imshow(transforms.ToPILImage()(original_images[col]))
        ax[0][col].set_title(f"Original: {original_labels[col]}")
        ax[0][col].axis("off")

    for col in range(6):
        ax[1][col].imshow(transforms.ToPILImage()(perturbed_images[col]))
        ax[1][col].set_title(f"Perturbed: {perturbed_labels[col]}")
        ax[1][col].axis("off")

    fig.suptitle("DeepFool attack on ResNet34")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.optim as optim
    import torch.utils.data as data_utils
    import math
    import torchvision.models as models
    from PIL import Image
    from deepfool.deepfool import deepfool, local_deepfool, deepfool_specific
    import os

    (
        original_images,
        original_labels,
        perturbed_images,
        perturbed_labels,
        max_pixel_values,
        diff_norms,
        region_masks,
    ) = make_examples()

    plot_diff(original_images, perturbed_images, region_masks)

    plot_comparaison(
        original_images, perturbed_images, original_labels, perturbed_labels
    )

    print(f"shape of original_images: {original_images[0].shape}")
    print(f"shape of perturbed_images: {perturbed_images[0].shape}")

    # # Display bar charts for max pixel values and norms of differences for each image
    # for i in range(5):
    #     fig, ax1 = plt.subplots(figsize=(10, 5))
    #     indices = range(16)
    #     max_vals = max_pixel_values[i]
    #     norm_vals = diff_norms[i]

    #     width = 0.35  # Width of the bars
    #     ax1.bar(indices, max_vals, width=width, label='Max Pixel Value', alpha=0.7, color='b')
    #     ax1.set_xlabel('Region Index')
    #     ax1.set_ylabel('Max Pixel Value', color='b')
    #     ax1.tick_params(axis='y', labelcolor='b')

    #     ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    #     ax2.bar([x + width for x in indices], norm_vals, width=width, label='Norm of Difference', alpha=0.7, color='r')
    #     ax2.set_ylabel('Norm of Difference', color='r')
    #     ax2.tick_params(axis='y', labelcolor='r')

    #     plt.title(f'Max Pixel Value and Norm of Difference for Image {i+1}')
    #     fig.tight_layout()
    #     plt.show()

    # # Print max pixel values and norms of differences
    # for i, (max_vals, norm_vals) in enumerate(zip(max_pixel_values, diff_norms)):
    #     for j, (max_val, norm_val) in enumerate(zip(max_vals, norm_vals)):
    #         print(f"Image {i+1}, Region {j+1}: Max Pixel Value = {max_val}, Norm of Difference = {norm_val}")
