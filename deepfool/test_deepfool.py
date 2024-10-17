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
from deepfool import deepfool
import os
from torchvision.models import ResNet34_Weights

# Load pretrained ResNet-34 model
net = models.resnet34(weights=ResNet34_Weights.DEFAULT)

# Switch to evaluation mode
net.eval()

# Load image
im_orig = Image.open('data/test_img2.jpg')

# Mean and std used for normalization (ImageNet stats)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Preprocessing the image: resize, crop, convert to tensor, and normalize
im = transforms.Compose([
    transforms.Resize(256),  # Updated from transforms.Scale
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])(im_orig)

# Run DeepFool attack
r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

# Load class labels from file
labels = open(os.path.join('data/synset_words.txt'), 'r').read().split('\n')

# Get original and perturbed class labels
str_label_orig = labels[int(label_orig)].split(',')[0]  # Changed np.int to int
str_label_pert = labels[int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)

# Function to clip tensor values between minv and maxv
def clip_tensor(A, minv, maxv):
    A = torch.clamp(A, minv, maxv)  # Use torch.clamp for cleaner implementation
    return A

# Clipping function for images (0-255 range)
clip = lambda x: clip_tensor(x, 0, 255)

# Inverse transformation to convert perturbed image back to PIL format
tf = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[1/s for s in std]),  # Reverse normalization
    transforms.Normalize(mean=[-m for m in mean], std=[1, 1, 1]),  # Subtract mean
    transforms.Lambda(clip),  # Clip the values to ensure valid image range
    transforms.ToPILImage(),  # Convert tensor back to PIL image
    transforms.CenterCrop(224)  # Center crop to 224x224
])

pert_image = pert_image.cpu().view(pert_image.size()[-3:]).type(torch.FloatTensor)

# Display the perturbed image
plt.figure()
print(f"Original image shape: {im.shape}")
print(f"Perturbed image shape: {pert_image.shape}")
# plt.imshow(tf(pert_image - im))
# print(pert_image - im)
plt.imshow(tf(pert_image))  # Ensure image is on CPU before converting to PIL
plt.title(str_label_pert)
plt.show()
