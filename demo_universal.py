import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import tarfile
import urllib.request
import sys
import getopt
from torchvision import models, transforms
from universal.prepare_imagenet_data import (
    preprocess_image_batch,
    create_imagenet_npy,
    undo_image_avg,
)
from universal.universal_pert import universal_perturbation

import matplotlib.pyplot as plt
from urllib.request import urlretrieve
import zipfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10


def jacobian(y, x, inds):
    jacobian = []
    for ind in inds:
        grad_output = torch.zeros_like(y)
        grad_output[0][ind] = 1
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=grad_output,
            retain_graph=True,
            create_graph=True,
        )[0]
        jacobian.append(gradients)
    return torch.stack(jacobian)


def show_dl(block_num, block_size, total_size):
    taille_recue = block_num * block_size
    if total_size > 0:
        pourcentage = min(100, taille_recue * 100 / total_size)
        sys.stdout.write(f"\rDownload : {pourcentage:.2f}%")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rReceived size : {taille_recue / (1024 * 1024):.2f} MB")
        sys.stdout.flush()


if __name__ == "__main__":
    # Parse arguments
    argv = sys.argv[1:]

    # Default values
    path_train_imagenet = "data/demo_universal/imagenette2-320/train"
    path_test_image = "data/demo_universal/test_img.jpg"

    try:
        opts, args = getopt.getopt(argv, "i:t:", ["test_image=", "training_path="])
    except getopt.GetoptError:
        print("python " + sys.argv[0] + " -i <test image> -t <imagenet training path>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-t":
            path_train_imagenet = arg
        if opt == "-i":
            path_test_image = arg

    # Download Inception model if not already present
    model_path = os.path.join("data", "demo_universal", "inception_v3.pth")
    if not os.path.isfile(model_path):
        print("Downloading Inception model...")
        urlretrieve(
            "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth",
            model_path,
            show_dl,
        )

    # Load dataset
    # Définir le chemin du dossier et l'URL de l'archive
    target_folder = "data/demo_universal/imagenette2-320"
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    archive_path = "data/demo_universal/imagenette2-320.tgz"

    # Vérifier si le dossier existe
    if not os.path.exists(target_folder):
        print(f"The folder '{target_folder}' doesn't exist. Dataset downloads...")

        # Créer le dossier parent s'il n'existe pas
        os.makedirs("data/demo_universal", exist_ok=True)

        # Télécharger l'archive
        urllib.request.urlretrieve(dataset_url, archive_path, show_dl)
        print("Téléchargement terminé.")

        # Extraire l'archive
        print("Extraction de l'archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path="data/demo_universal")
        print("Extraction terminée.")

        # Optionnel : supprimer l'archive après extraction
        os.remove(archive_path)
        print("Archive supprimée.")
    else:
        print(
            f"Le dossier '{target_folder}' existe déjà. Aucun téléchargement nécessaire."
        )

    # Load the Inception model and set it to evaluation mode
    model = models.inception_v3(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    print(">> Computing feedforward function...")

    def f(image_inp):
        # Convert to pytorch format
        image_inp = np.transpose(image_inp, (0, 3, 1, 2))
        with torch.no_grad():
            image_inp = torch.tensor(image_inp).to(device)
            output = model(image_inp)
        return output.cpu().numpy()

    file_perturbation = os.path.join("data", "demo_universal", "universal.npy")

    if not os.path.isfile(file_perturbation):
        print(
            ">> Compiling the gradient PyTorch functions. This might take some time..."
        )

        # Compute the gradient function
        def grad_fs(image_inp, indices):
            image_inp = torch.tensor(image_inp, requires_grad=True, device=device)
            output = model(image_inp)
            return jacobian(output, image_inp, indices).cpu().numpy()

        # Load/Create data
        datafile = os.path.join("data", "demo_universal", "imagenet_data.npy")
        if not os.path.isfile(datafile):
            print(">> Creating pre-processed imagenet data...")
            X = create_imagenet_npy(path_train_imagenet)

            print(">> Saving the pre-processed imagenet data")
            if not os.path.exists("data"):
                os.makedirs("data")
            np.save(datafile, X)
        else:
            print(">> Pre-processed imagenet data detected")
            X = np.load(datafile)

        # Running universal perturbation
        v = universal_perturbation(
            X, model, grad_fs, delta=0.2, num_classes=num_classes
        )

        # Saving the universal perturbation
        np.save(file_perturbation, v)
    else:
        print(
            f">> Found a pre-computed universal perturbation! Retrieving it from {file_perturbation}"
        )
        v = np.load(file_perturbation)

    print(">> Testing the universal perturbation on an image")

    # Test the perturbation on the image
    with open(os.path.join("data", "demo_universal", "labels.txt"), "r") as f:
        labels = f.read().split("\n")

    image_original = preprocess_image_batch(
        [path_test_image], img_size=(256, 256), crop_size=(224, 224), color_mode="rgb"
    )
    label_original = np.argmax(f(image_original), axis=1).flatten()
    str_label_original = labels[int(label_original) - 1].split(",")[0]

    # Clip the perturbation to make sure images fit in uint8
    clipped_v = np.clip(undo_image_avg(image_original[0] + v[0]), 0, 255) - np.clip(
        undo_image_avg(image_original[0]), 0, 255
    )

    image_perturbed = image_original + clipped_v[None, :, :, :]
    label_perturbed = np.argmax(f(image_perturbed), axis=1).flatten()
    str_label_perturbed = labels[int(label_perturbed) - 1].split(",")[0]

    # Show original and perturbed image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(undo_image_avg(image_original[0]).astype("uint8"), interpolation=None)
    plt.title(str_label_original)

    plt.subplot(1, 2, 2)
    plt.imshow(undo_image_avg(image_perturbed[0]).astype("uint8"), interpolation=None)
    plt.title(str_label_perturbed)

    plt.show()
