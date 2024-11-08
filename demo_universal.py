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

# Transformation pour le dataset STL-10
transform = transforms.Compose(
    [
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Télécharger et charger le dataset STL-10 pour l'entraînement
train_set = torchvision.datasets.STL10(
    root="./data",
    split="train",
    download=True,
    transform=transform,
)

"""# Limiter à 100 images par classe
class_counts = defaultdict(int)
max_per_class = 10
filtered_indices = []

for idx, (_, label) in enumerate(train_set):
    if class_counts[label] < max_per_class:
        filtered_indices.append(idx)
        class_counts[label] += 1

# Création du sous-ensemble personnalisé
train_subset = torch.utils.data.Subset(train_set, filtered_indices)"""

# Création du DataLoader avec le sous-ensemble
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=10,
    shuffle=True,
    num_workers=2,
)

# Télécharger et charger le dataset STL-10 pour les tests
test_set = torchvision.datasets.STL10(
    root="./data",
    split="test",
    download=True,
    transform=transform,
)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=4, shuffle=False, num_workers=2
)

# Classes du dataset STL-10
classes = (
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
)

# Charger le modèle ResNet-18
model = models.resnet18(pretrained=True)

# Adapter le modèle pour STL-10 (10 classes au lieu de 1000)
model.fc = nn.Linear(model.fc.in_features, len(classes))

# Définir l'appareil (GPU si disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Fonction pour afficher les images
def imshow(img, label=""):
    npimg = img.detach().cpu().numpy()
    if npimg.shape[0] == 3:
        npimg = np.transpose(npimg, (1, 2, 0))
    npimg = (npimg - np.min(npimg)) / (np.max(npimg) - np.min(npimg))
    npimg = (npimg * 255).astype(np.uint8)
    plt.imshow(npimg)
    plt.title(label)
    plt.axis("off")


# Fonction pour charger une image individuelle
def load_image(image_path, size=(96, 96)):
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)


# Génération de la perturbation universelle
v = universal_perturbation(
    train_loader, model, v_size=96, device=device, delta=0.95, num_classes=len(classes)
)

# Charger et tester une image
test_img = load_image("data/demo_universal/test_img.jpg").to(device)

test_img_label = classes[int(torch.argmax(model(test_img)).item())]

perturbed_test_img = test_img + v

perturbed_test_img_label = classes[int(torch.argmax(model(perturbed_test_img)).item())]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
imshow(test_img.squeeze(), test_img_label)

plt.subplot(1, 3, 2)
imshow(perturbed_test_img.squeeze(), perturbed_test_img_label)

plt.subplot(1, 3, 3)
imshow(v.squeeze(), "Universal perturbation")

plt.show()
