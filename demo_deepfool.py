def make_examples():
    net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    net.eval()

    original_images = []
    original_labels = []
    perturbed_images = []
    perturbed_labels = []
    max_pixel_values = []
    diff_norms = []

    for i in range(1, 6):
        im_orig = Image.open(f"data/demo_deepfool/test_img{i}.jpg")

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        original_images.append(
            transforms.Compose(
                [
                    transforms.Resize(256),  
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )(im_orig)
        )

        im = transforms.Compose(
            [
                transforms.Resize(256), 
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )(im_orig).cpu()  
        
        input_shape = im.cpu().numpy().shape
        print(input_shape[1:])
        region_mask = np.zeros(input_shape[1:], dtype=np.int32)
        region_mask[50:150, 50:150] = 1

       
        r, loop_i, label_orig, label_pert, pert_image = local_deepfool(
            im, net, max_iter=1000, region_mask=region_mask
        )

        # PARTIE UTILISEE POUR PLOT LES VALEURS DE PIXELS ET NORMES EN FONCTION DES REGIONS CHOISIES
        # # Define the regions for local DeepFool (4x4 grid)
        # height, width = im.shape[1], im.shape[2]
        # regions = []
        # grid_size = 4
        # step_h = height // grid_size
        # step_w = width // grid_size

        # for row in range(grid_size):
        #     for col in range(grid_size):
        #         top = row * step_h
        #         left = col * step_w
        #         bottom = top + step_h
        #         right = left + step_w
        #         regions.append((top, left, bottom, right))

        # # Apply local DeepFool to each region
        # image_max_pixel_values = []
        # image_diff_norms = []
        # for region in regions:
        #     r, loop_i, label_orig, label_pert, pert_image = local_deepfool(
        #         im, net, max_iter=1000, region=region
        #     )

        #     # Ensure perturbed image is on the CPU
        #     pert_image = pert_image.cpu()

        #     # Calculate the difference
        #     diff_tensor = torch.abs(im - pert_image)
        #     diff_gray = torch.mean(diff_tensor, dim=0)  # Convert to grayscale by averaging channels

        #     # Store the max pixel value and norm of the difference
        #     image_max_pixel_values.append(diff_gray.max().item())
        #     image_diff_norms.append(torch.norm(diff_gray).item())

        # max_pixel_values.append(image_max_pixel_values)
        # diff_norms.append(image_diff_norms)

        # Load class labels from file
        labels = (
            open(os.path.join("data/demo_deepfool/synset_words.txt"), "r")
            .read()
            .split("\n")
        )

        str_label_orig = labels[int(label_orig)].split(",")[0]  
        str_label_pert = labels[int(label_pert)].split(",")[0]

        original_labels.append(str_label_orig.split()[1:])
        perturbed_labels.append(str_label_pert.split()[1:])

        def clip_tensor(A, minv, maxv):
            A = torch.clamp(A, minv, maxv)  
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

        pert_image = (
            pert_image.view(pert_image.size()[-3:]).type(torch.FloatTensor)
        )
        perturbed_images.append(tf(pert_image))
    return original_images, original_labels, perturbed_images, perturbed_labels, max_pixel_values, diff_norms


def diff(original_images, perturbed_images):
    difference_images = []

    for orig, pert in zip(original_images, perturbed_images):
        if isinstance(orig, Image.Image):
            orig_tensor = transforms.ToTensor()(orig)
        else:
            orig_tensor = orig
        if isinstance(pert, Image.Image):
            pert_tensor = transforms.ToTensor()(pert)
        else:
            pert_tensor = pert

        diff_tensor = torch.abs(orig_tensor - pert_tensor)

        diff_image = transforms.ToPILImage()(diff_tensor)
        difference_images.append(diff_image)
        
    return difference_images


def plot_diff(original_images, perturbed_images):
    difference_images = diff(original_images, perturbed_images)

    fig_diff, ax_diff = plt.subplots(1, 5, figsize=(20, 5))
    for col in range(5):
        diff_im = transforms.ToTensor()(difference_images[col])
        diff_gray = torch.mean(diff_im, dim=0)  # Convert to grayscale by averaging channels
        im = ax_diff[col].imshow(diff_gray, cmap="gray")
        ax_diff[col].set_title(f"Difference {col+1}")
        ax_diff[col].axis("off")
        fig_diff.colorbar(im, ax=ax_diff[col], orientation='vertical')

    fig_diff.suptitle("Difference between Original and Perturbed Images")
    plt.tight_layout()
    plt.show()
    
def plot_comparaison(original_images, perturbed_images, original_labels, perturbed_labels):
    fig, ax = plt.subplots(2, 5, figsize=(12, 8))
    for col in range(5):
        ax[0][col].imshow(transforms.ToPILImage()(original_images[col]))
        ax[0][col].set_title(f"Original: {original_labels[col]}")
        ax[0][col].axis("off")

    for col in range(5):
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
    from deepfool.deepfool_yolo import deepfool, local_deepfool
    import os
    

    original_images, original_labels, perturbed_images, perturbed_labels, max_pixel_values, diff_norms = make_examples()

    # plot_diff(original_images, perturbed_images)
    
    # plot_comparaison(original_images, perturbed_images, original_labels, perturbed_labels)

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