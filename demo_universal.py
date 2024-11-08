import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from universal.universal_pert import universal_perturbation
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from PIL import Image

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Télécharger et charger le dataset CIFAR-10 pour l'entraînement
train_set = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

# Limiter à 100 images par classe
class_counts = defaultdict(int)
max_per_class = 10
filtered_indices = []

# Parcours du dataset pour sélectionner les indices
for idx, (_, label) in enumerate(train_set):
    if class_counts[label] < max_per_class:
        filtered_indices.append(idx)
        class_counts[label] += 1

# Création du sous-ensemble personnalisé
train_subset = torch.utils.data.Subset(train_set, filtered_indices)

# Création du DataLoader avec le sous-ensemble
train_loader = torch.utils.data.DataLoader(
    train_subset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
)

# Télécharger et charger le dataset CIFAR-10 pour les tests
test_set = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=4, shuffle=False, num_workers=2
)

# Classes du dataset CIFAR-10
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Charger le modèle de classification Inception v3
model = models.inception_v3(pretrained=True)

# Adapter le modèle pour CIFAR-10 (10 classes au lieu de 1000)
model.fc = nn.Linear(model.fc.in_features, 10)

# Définir l'appareil (GPU si disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def imshow(img):
    # Denormalize the image to [0, 1] range
    img = img / 2 + 0.5  # Assuming the image was normalized in the range [-1, 1]

    # Convert to numpy array and transpose to (H, W, C)
    npimg = img.cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)

    plt.imshow(npimg)
    plt.axis("off")


def load_image(image_path, size=(299, 299)):
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)


v = universal_perturbation(
    train_loader, model, device, delta=0.95, num_classes=len(classes)
)

test_img = load_image("data/demo_universal/test_img.jpg").to(device)

perturbed_test_img = test_img + v

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
imshow(test_img.squeeze())

plt.subplot(1, 3, 2)
imshow(perturbed_test_img.squeeze())

plt.subplot(1, 3, 3)
imshow(v.squeeze())

plt.show()
