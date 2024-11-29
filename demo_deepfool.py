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

        # Preprocessing the image: resize, crop, convert to tensor, and normalize
        im = transforms.Compose(
            [
                transforms.Resize(256),  # Updated from transforms.Scale
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )(im_orig).cpu()  # Ensure the tensor is on the CPU
        
        # Creating the region mask
        input_shape = im.cpu().numpy().shape
        print(input_shape[1:])
        region_mask = np.zeros(input_shape[1:], dtype=np.int32)
        region_mask[50:150, 50:150] = 1

        # plt.imshow((region_mask))
        # plt.imshow(region_mask, cmap='gray')
        # plt.show()

        # Run DeepFool attack
        # r, loop_i, label_orig, label_pert, pert_image = local_deepfool(
        #    im, net, max_iter=1000, region_mask=region_mask
        # )
        #DEMO DEEPFOOL SPECIFIC

        r, loop_i, label_orig, label_pert, pert_image = deepfool_specific(
           im, net, 412, max_iter=1000)


        # PARTIE UTILISEE POUR PLOT LES VALEURS DE PIXELS ET NORMES EN FONCTION DES REGIONS CHOISIES
        # Define the regions for local DeepFool (4x4 grid)
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
        #         regions.append([left, right, top, bottom])

        # # Apply local DeepFool to each region
        # image_max_pixel_values = []
        # image_diff_norms = []
        # for region in regions:
        #     region_mask = np.zeros(input_shape[1:], dtype=np.int32)
        #     region_mask[region[0]:region[1], region[2]:region[3]] = 1
            
        #     r, loop_i, label_orig, label_pert, pert_image = local_deepfool(
        #         im, net, max_iter=1000, region_mask=region_mask
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

        pert_image = (
            pert_image.view(pert_image.size()[-3:]).type(torch.FloatTensor)
        )
        perturbed_images.append(tf(pert_image))
    return original_images, original_labels, perturbed_images, perturbed_labels, max_pixel_values, diff_norms


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


def plot_diff(original_images, perturbed_images):
    difference_images = diff(original_images, perturbed_images)

    # Display the difference images with colorbars
    fig_diff, ax_diff = plt.subplots(1, 6, figsize=(20, 5))
    for col in range(6):
        # Convert the difference tensor to grayscale for visualization
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

def iterate_images_in_tree(root_dir, process_image_callback):
    """
    Parcourt récursivement un dossier et charge les images pour les traiter.

    :param root_dir: Répertoire racine à parcourir.
    :param process_image_callback: Fonction à appeler pour traiter chaque image.
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                file_path = os.path.join(subdir, file)
                try:
                    # Charger l'image
                    with Image.open(file_path) as img:
                        print(f"Traitement de l'image : {file_path}")
                        process_image_callback(img)
                except Exception as e:
                     print(f"Erreur lors du chargement de l'image {file_path}: {e}")   
                     
                              
def test_dataset():
    # Load pretrained ResNet-34 model
    net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    # Switch to evaluation mode
    net.eval()

    original_labels = []
    perturbed_labels = []
    max_pixel_values = []
    diff_norms = []

    #for subdir, _, files in os.walk("data/demo_deepfool/imagenette2-320/train/mini_train_imagenette"):
    for subdir, _, files in os.walk("data/demo_deepfool/imagenette2-320/train"):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                file_path = os.path.join(subdir, file)
                try:
                    # Charger l'image
                    with Image.open(file_path) as im_orig:
                        print(f"Traitement de l'image : {file_path}")
                        # Mean and std used for normalization (ImageNet stats)
                        mean = [0.485, 0.456, 0.406]
                        std = [0.229, 0.224, 0.225]

                        # Preprocessing the image: resize, crop, convert to tensor, and normalize
                        im = transforms.Compose(
                            [
                                transforms.Resize(256),  # Updated from transforms.Scale
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std),
                            ]
                        )(im_orig).cpu()  # Ensure the tensor is on the CPU

                        r, loop_i, label_orig, label_pert, pert_image = local_deepfool(
                        im, net, max_iter=1000
                        )

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

                except Exception as e:
                     print(f"Erreur lors du chargement de l'image {file_path}: {e}")  

        
    return original_labels, perturbed_labels, max_pixel_values, diff_norms



    
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter

def display_readable_confusion_matrix(original_labels, perturbed_labels, top_n_classes=30, output_path="confusion_matrix_readable.png"):
    """
    Affiche une matrice de confusion lisible avec les N classes les plus fréquentes.
    Affiche et enregistre également les valeurs de la diagonale avec les classes correspondantes.

    :param original_labels: Liste des étiquettes originales.
    :param perturbed_labels: Liste des étiquettes perturbées.
    :param top_n_classes: Nombre maximum de classes à afficher dans la matrice.
    :param output_path: Chemin où enregistrer l'image de la matrice de confusion.
    """
    # Aplatir les listes si elles contiennent des sous-listes
    original_labels = [" ".join(label) for label in original_labels]
    perturbed_labels = [" ".join(label) for label in perturbed_labels]

    # Compter les occurrences de chaque classe pour sélectionner les plus fréquentes
    label_counts = Counter(original_labels)
    most_common_classes = [label for label, _ in label_counts.most_common(top_n_classes)]

    filtered_original = []
    filtered_perturbed = []

    for orig, pert in zip(original_labels, perturbed_labels):
        if orig in most_common_classes and pert in most_common_classes:
            filtered_original.append(orig)
            filtered_perturbed.append(pert)

    # Vérifier que les longueurs sont cohérentes
    if len(filtered_original) != len(filtered_perturbed):
        raise ValueError("Les listes filtrées ne sont toujours pas alignées.")

    # Calculer la matrice de confusion
    cm = confusion_matrix(filtered_original, filtered_perturbed, labels=most_common_classes)

    # Affichage de la matrice de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=most_common_classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')

    plt.title(f"Matrice de confusion ({top_n_classes} classes les plus fréquentes)")
    plt.savefig(output_path, bbox_inches="tight", dpi=300) 
    plt.show()


    print("\nValeurs de la diagonale (prédictions correctes) :")
    for i, label in enumerate(most_common_classes):
        print(f"Classe: {label}, Prédictions correctes: {cm[i, i]}")





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
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    

    # original_images, original_labels, perturbed_images, perturbed_labels, max_pixel_values, diff_norms = make_examples()

    # plot_diff(original_images, perturbed_images)
    
    # plot_comparaison(original_images, perturbed_images, original_labels, perturbed_labels)

    # print(f"shape of original_images: {original_images[0].shape}")
    # print(f"shape of perturbed_images: {perturbed_images[0].shape}")
    
    # Appel de la fonction principale
    original_labels, perturbed_labels, max_pixel_values, diff_norms = test_dataset()


    # Affichage et enregistrement de la heatmap
    display_readable_confusion_matrix(original_labels, perturbed_labels, output_path="confusion_matrix_local_deepfool.png")


    

    # # Display bar charts for max pixel values and norms of differences for each image
    # for i in range(5):
    #     fig, ax1 = plt.subplots(figsize=(10, 5))
    #     indices = range(16)
    #     max_vals = max_pixel_values[i]
    #     norm_vals = diff_norms[i]
        
    #     width = 0.35  # Width of the bars
    #     ax1.bar(indices, max_vals, width=width, label='Max Pixel Value', alpha=0.7, color='b')
    #     ax1.set_xlabel('Region Index')
    #     ax1.set_ylabel('Max Pixel Value', color='b')"data/demo_deepfool/test_img{i}.jpg")
    # # Print max pixel values and norms of differences
    # for i, (max_vals, norm_vals) in enumerate(zip(max_pixel_values, diff_norms)):
    #     for j, (max_val, norm_val) in enumerate(zip(max_vals, norm_vals)):
    #         print(f"Image {i+1}, Region {j+1}: Max Pixel Value = {max_val}, Norm of Difference = {norm_val}")
            
    
    





