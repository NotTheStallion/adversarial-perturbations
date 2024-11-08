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
from deepfool.deepfool import deepfool, local_deepfool
import os


def make_examples():
    # Load pretrained ResNet-34 model
    net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    # Switch to evaluation mode
    net.eval()

    original_images = []
    original_labels = []
    perturbed_images = []
    perturbed_labels = []
    perturbation = []

    for i in range(1, 6):
        # Load image
        im_orig = Image.open(f"data/demo_deepfool/test_img{i}.jpg")

        # Mean and std used for normalization (ImageNet stats)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]


        original_images.append(transforms.Compose(
            [
                transforms.Resize(256),  # Updated from transforms.Scale
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ]
        )(im_orig))

        # Preprocessing the image: resize, crop, convert to tensor, and normalize
        im = transforms.Compose(
            [
                transforms.Resize(256),  # Updated from transforms.Scale
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )(im_orig)
        
        # Run DeepFool attack
        r, loop_i, label_orig, label_pert, pert_image = local_deepfool(im, net, max_iter=1000, region=(0, 0, 150, 150))

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
            ]
        )

        pert_image = (
            pert_image.cpu().view(pert_image.size()[-3:]).type(torch.FloatTensor)
        )
        
        original_images.append(tf(im))
        perturbed_images.append(tf(pert_image))
    return original_images, original_labels, perturbed_images, perturbed_labels


original_images, original_labels, perturbed_images, perturbed_labels = make_examples()
# Calculate and display the difference between original and perturbed images
difference_images = []

for orig, pert in zip(original_images, perturbed_images):
    print(type(orig), type(pert))
    
    # Convert images to tensors
    if isinstance(orig, Image.Image): 
        orig_tensor = transforms.ToTensor()(orig)
    else :
        orig_tensor = orig
    if isinstance(pert, Image.Image): 
        pert_tensor = transforms.ToTensor()(pert)
    else:
        pert_tensor = pert
    
    
    print(type(orig_tensor), type(pert_tensor))
    
    # Calculate the difference
    diff_tensor = torch.abs(orig_tensor - pert_tensor)
    
    # Convert the difference tensor back to a PIL image
    diff_image = transforms.ToPILImage()(diff_tensor)
    difference_images.append(diff_image)

# print(transforms.ToTensor()(difference_images[-1]))

# Display the difference images
fig_diff, ax_diff = plt.subplots(1, 5, figsize=(12, 4))
for col in range(5):
    # Rescale the difference image to the range [0, 1] for better visualization
    diff = transforms.ToTensor()(difference_images[col])
    diff_rescaled = transforms.ToPILImage()(diff/diff.max())
    ax_diff[col].imshow(diff_rescaled, cmap='gray')
    ax_diff[col].set_title(f"Difference {col+1}")
    ax_diff[col].axis("off")

fig_diff.suptitle("Difference between Original and Perturbed Images")
plt.tight_layout()
plt.show()
print(f"shape of original_images: {original_images[0].shape}")
print(f"shape of perturbed_images: {transforms.ToTensor()(perturbed_images[0]).shape}")

fig, ax = plt.subplots(3, 5, figsize=(12, 8))
for col in range(5):
    ax[0][col].imshow(transforms.ToPILImage()(original_images[col]))
    ax[0][col].set_title(f"Original: {original_labels[col]}")
    ax[0][col].axis("off")
    
    ax[1][col].imshow(perturbation[col].transpose(2, 1, 0) * 255)
    ax[1][col].set_title(f"Perturbation added")
    ax[1][col].axis("off")

    ax[2][col].imshow(perturbed_images[col])
    ax[2][col].set_title(f"Perturbed image: {perturbed_labels[col]}")
    ax[2][col].axis("off")

fig.suptitle("DeepFool attack on ResNet34")
plt.tight_layout()
plt.show()
