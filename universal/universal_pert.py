import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from universal.deepfool import deepfool
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor
import torch.nn.functional as F

try:
    import wandb
    wandb_enabled = True
except ImportError:
    wandb_enabled = False

wandb_enabled = False

def proj_lp(v, xi, p):
    if p == 2:
        v = v * min(1, xi / torch.norm(v))
    elif p == float("inf"):
        v = torch.sign(v) * torch.minimum(v.abs(), torch.tensor(xi, device=v.device))
    else:
        raise ValueError(
            "Values of p different from 2 and Inf are currently not supported..."
        )
    return v


def universal_perturbation(
    dataloader,
    testloader,
    f,
    v_size,
    device,
    delta=0.2,
    max_iter_uni=np.inf,
    xi=10,
    p=float("inf"),
    num_classes=10,
    overshoot=0.02,
    max_iter_df=100,
):
    if wandb_enabled:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="Universal Perturbation",
            # track hyperparameters and run metadata
            config={
                "model": type(f).__name__,
                "dataset": "STL-10",
                "delta": delta,
                "xi": xi,
            },
        )

    # Set model to evaluation mode
    f.eval()

    v = torch.zeros(3, v_size, v_size).to(device)
    fooling_rate = 0.0
    itr_count = 0

    print("Start the computation of the universal perturbation")
    print(f"Parameters: delta={delta} ({1 - delta} fooling rate), xi={xi}, p={p}")

    while fooling_rate < 1 - delta and itr_count < max_iter_uni:
        print("Starting pass number ", itr_count)

        for batch_idx, (images, _) in enumerate(dataloader):
            print(
                f"\r[...] Processing batch {batch_idx + 1} on {len(dataloader)} (batch size: {len(images)})",
                end="",
            )
            images = images.to(device)

            for img in images:
                v = v.to(device)

                img = img.unsqueeze(0)

                if int(torch.argmax(f(img)).item()) == int(
                    torch.argmax(f(img + v)).item()
                ):
                    perturbation, num_iterations, *_ = deepfool(
                        (img + v).squeeze(0),
                        f,
                        num_classes=num_classes,
                        overshoot=overshoot,
                        max_iter=max_iter_df,
                    )

                    perturbation = torch.from_numpy(perturbation).to(device).float()

                    if num_iterations < max_iter_df - 1:
                        v = v + perturbation
                        v = proj_lp(v, xi, p)

        itr_count += 1

        # Compute the fooling rate
        est_labels_orig = []
        est_labels_pert = []

        with torch.no_grad():
            for images, _ in testloader:
                images = (
                    F.interpolate(
                        images,
                        size=(v_size, v_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .to(device)
                )
                perturbed_images = images + v

                est_labels_orig.extend(torch.argmax(f(images), dim=1).cpu().numpy())
                est_labels_pert.extend(
                    torch.argmax(f(perturbed_images), dim=1).cpu().numpy()
                )

        fooling_rate = np.mean(np.array(est_labels_orig) != np.array(est_labels_pert))
        print(f"\rFOOLING RATE = {fooling_rate}\033[K")

    if wandb_enabled:
        wandb.finish()
    return v
