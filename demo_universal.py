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

"""
# Exemple de visualisation d'un batch de données

def imshow(img):
    img = img / 2 + 0.5  # Dénormalise l'image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Obtenir un batch d'images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Afficher les images
imshow(torchvision.utils.make_grid(images))
# Afficher les labels
print(" ".join(f"{classes[labels[j]]}" for j in range(4)))"""

v = universal_perturbation(
    train_loader, model, device, delta=0.2, num_classes=len(classes)
)
