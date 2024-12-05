import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from universal.universal_pert import universal_perturbation
import matplotlib.pyplot as plt
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

# Création du DataLoader pour l'entraînement
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=100,
    shuffle=True,
)

# Télécharger et charger le dataset STL-10 pour les tests
test_set = torchvision.datasets.STL10(
    root="./data",
    split="test",
    download=True,
    transform=transform,
)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

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


# Custom CNN
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # Convolution 1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # Convolution 2
        self.fc1 = nn.Linear(64 * 22 * 22, 128)  # Fully connected 1
        self.fc2 = nn.Linear(128, 10)  # Fully connected 2 (10 classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # Pooling 1
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)  # Pooling 2

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))  # Fully connected 1
        x = self.fc2(x)  # Fully connected 2
        return x


# Vérification dynamique des dimensions
dummy_input = torch.randn(1, 3, 96, 96)  # Exemple d'entrée
model_test = CustomCNN()

# Déplacement du modèle sur GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = CustomCNN().to(device)

# Définir l'optimiseur et la fonction de perte
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Taux d'apprentissage ajusté
criterion = nn.CrossEntropyLoss()

# Boucle d'entraînement
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Mode entraînement
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Réinitialisation des gradients

        outputs = model(inputs)  # Passage avant
        loss = criterion(outputs, labels)  # Calcul de la perte

        loss.backward()  # Rétropropagation
        optimizer.step()  # Mise à jour des poids

        running_loss += loss.item()

    # Évaluation sur le dataset de test
    model.eval()  # Mode évaluation
    total = 0
    correct = 0
    with torch.no_grad():  # Pas de calcul des gradients
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Affichage des résultats par époque
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%"
    )

print("Entraînement terminé.")

# Évaluation finale
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Précision finale sur le dataset de test : {100 * correct / total:.2f}%")


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")


def imshow(img, label=""):
    npimg = img.detach().cpu().numpy()
    if npimg.shape[0] == 3:
        npimg = np.transpose(npimg, (1, 2, 0))
    npimg = (npimg - np.min(npimg)) / (np.max(npimg) - np.min(npimg))
    npimg = (npimg * 255).astype(np.uint8)
    plt.imshow(npimg)
    plt.title(label)
    plt.axis("off")


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
    train_loader,
    model,
    v_size=96,
    device=device,
    delta=0.2,
    xi=0.5,
    p=float("inf"), #float("inf") or 2
    num_classes=len(classes),
)

print(f"Perturbation universelle générée: norme L2 = {torch.norm(v).item()}, norme L∞ = {torch.max(torch.abs(v)).item()}")

# Sauvegarde de la perturbation universelle
torch.save(v, "universal_perturbation_stl10_resnet18.pth")

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
