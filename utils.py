import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision import models
import os
from tqdm import tqdm


def diff(original_images, perturbed_images):
    difference_images = []
    for orig, pert in zip(original_images, perturbed_images):
        orig_tensor = (
            transforms.ToTensor()(orig) if isinstance(orig, Image.Image) else orig
        )
        pert_tensor = (
            transforms.ToTensor()(pert) if isinstance(pert, Image.Image) else pert
        )
        diff_tensor = torch.abs(orig_tensor - pert_tensor)
        diff_image = transforms.ToPILImage()(diff_tensor)
        difference_images.append(diff_image)
    return difference_images


def plot_diff(original_images_norm, perturbed_images_norm):
    assert (
        len(original_images_norm) == len(perturbed_images_norm)
    ), f"Number of original ({len(original_images_norm)}) and perturbed ({len(perturbed_images_norm)}) images must be the same"

    difference_images = diff(original_images_norm, perturbed_images_norm)
    fig_diff, ax_diff = plt.subplots(1, len(difference_images), figsize=(20, 5))
    for col in range(len(difference_images)):
        diff_im = transforms.ToTensor()(difference_images[col])
        diff_gray = torch.mean(diff_im, dim=0)
        im = ax_diff[col].imshow(diff_gray, cmap="gray")
        ax_diff[col].set_title(f"Difference {col+1}")
        ax_diff[col].axis("off")
        fig_diff.colorbar(im, ax=ax_diff[col], orientation="vertical")
    plt.tight_layout()
    plt.show()


def plot_comparaison(
    original_images, perturbed_images, original_labels, perturbed_labels
):
    assert (
        len(original_images)
        == len(perturbed_images)
        == len(original_labels)
        == len(perturbed_labels)
    ), f"Lengths are not equal: original_images={len(original_images)}, perturbed_images={len(perturbed_images)}, original_labels={len(original_labels)}, perturbed_labels={len(perturbed_labels)}"

    fig, ax = plt.subplots(2, len(original_images), figsize=(18, 7))
    for col in range(len(original_images)):
        ax[0][col].imshow(transforms.ToPILImage()(original_images[col]))
        ax[0][col].set_title(f"Original: {original_labels[col]}", fontsize=10)
        ax[0][col].axis("off")
        
        ax[1][col].imshow(transforms.ToPILImage()(perturbed_images[col]))
        ax[1][col].set_title(f"Perturbed: {perturbed_labels[col]}", fontsize=10)
        ax[1][col].axis("off")
    
    fig.suptitle("DeepFool Attack on ResNet34", fontsize=16)
    plt.subplots_adjust(top=0.85, hspace=0.3)
    plt.tight_layout()
    plt.show()


def make_examples(func, xargs):
    # Load pretrained ResNet-34 model
    net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    # Switch to evaluation mode
    net.eval()

    original_images = []
    original_labels = []
    original_images_norm = []
    perturbed_images = []
    perturbed_labels = []
    perturbed_images_norm = []

    for i in range(1, 7):
        # Load image
        im_orig = Image.open(f"data/demo_deepfool/test_img{i}.jpg")

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
        
        original_images_norm.append(
            transforms.Compose(
                [
                    transforms.Resize(256),  # Updated from transforms.Scale
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
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

        # Run the provided function (e.g., DeepFool attack)
        r, loop_i, label_orig, label_pert, pert_image = func(im, net, **xargs)

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
        def clip(x):
            return clip_tensor(x, 0, 255)

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
        perturbed_images_norm.append(pert_image)
    return (
        original_images,
        original_labels,
        original_images_norm,
        perturbed_images,
        perturbed_labels,
        perturbed_images_norm
    )


def perturb_set(perturb_func, data_set_loader, model):
    perturbed_images = []
    gt_labels = []
    
    for image, label in tqdm(data_set_loader, desc="Perturbing images"):
        r, loop_i, label_orig, label_pert, pert_image = perturb_func(image, model, num_classes=10, overshoot=0.02, max_iter=50, region_mask=None, verbose=False)
        perturbed_images.append(pert_image)
        gt_labels.append([label, label_orig, label_pert])
        # print(f"Original label: {label_orig}, Perturbed label: {label_pert}")
    
    perturbed_dataset = torch.utils.data.TensorDataset(torch.stack(perturbed_images), torch.tensor(gt_labels))
    # print(f"shape of perturbed dataset: {perturbed_dataset.tensors[1].shape}")
    return perturbed_dataset