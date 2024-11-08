import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from .deepfool_universal import deepfool
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor
import torch.nn.functional as F


def proj_lp(v, xi, p):
    if p == 2:
        v = v * min(1, xi / torch.norm(v.flatten(1)))
    elif p == float("inf"):
        v = torch.sign(v) * torch.minimum(v.abs(), torch.tensor(xi, device=v.device))
    else:
        raise ValueError(
            "Values of p different from 2 and Inf are currently not supported..."
        )
    return v


# Convertir un tensor en PIL image avant d'appliquer Resize
def resize_tensor(img_tensor, size=299):
    to_pil = ToPILImage()
    img_pil = to_pil(img_tensor)
    resize_transform = transforms.Resize(size)
    img_resized = resize_transform(img_pil)
    to_tensor = ToTensor()
    img_resized_tensor = to_tensor(img_resized)
    return img_resized_tensor


def universal_perturbation(
    dataloader,
    f,
    device,
    delta=0.2,
    max_iter_uni=np.inf,
    xi=10,
    p=float("inf"),
    num_classes=10,
    overshoot=0.02,
    max_iter_df=10,
):
    # Set model to evaluation mode
    f.eval()

    # Resize transform to ensure images are the correct size
    transform_resize = transforms.Compose(
        [
            transforms.Resize(299),  # Resize images to 299x299 for Inception model
            transforms.ToTensor(),
        ]
    )

    v = torch.zeros_like(next(iter(dataloader))[0][0]).to(device)
    fooling_rate = 0.0
    itr_count = 0

    while fooling_rate < 1 - delta and itr_count < max_iter_uni:
        print("Starting pass number ", itr_count)

        for images, _ in dataloader:
            images = images.to(device)

            for img in images:
                # Apply the resize transformation before passing to the model
                img_resized = resize_tensor(img, 299).to(device)

                img_resized = img_resized.unsqueeze(0)

                v = F.interpolate(
                    v.unsqueeze(0),
                    size=(299, 299),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                v = v.to(device)

                with torch.no_grad():  # Disable gradient tracking
                    if int(torch.argmax(f(img_resized)).item()) == int(
                        torch.argmax(f(img_resized + v)).item()
                    ):
                        print(">> Processing image...")

                        perturbation, num_iterations, _, _ = deepfool(
                            img_resized + v,
                            f,
                            num_classes=num_classes,
                            overshoot=overshoot,
                            max_iter=max_iter_df,
                        )

                        if num_iterations < max_iter_df - 1:
                            v = v + perturbation
                            v = proj_lp(v, xi, p)

        itr_count += 1

        # Compute the fooling rate
        est_labels_orig = []
        est_labels_pert = []

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                perturbed_images = images + v

                # Resize the images again before inference
                images_resized = transform_resize(images)
                perturbed_images_resized = transform_resize(perturbed_images)

                est_labels_orig.extend(
                    torch.argmax(f(images_resized), dim=1).cpu().numpy()
                )
                est_labels_pert.extend(
                    torch.argmax(f(perturbed_images_resized), dim=1).cpu().numpy()
                )

        fooling_rate = np.mean(np.array(est_labels_orig) != np.array(est_labels_pert))
        print("FOOLING RATE = ", fooling_rate)

    return v
