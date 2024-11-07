import numpy as np
import torch
from deepfool.deepfool import deepfool


def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi

    # Supports only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi / torch.norm(v.flatten(1)))  # Using torch norm
    elif p == np.inf:
        v = torch.sign(v) * torch.minimum(torch.abs(v), xi)  # Using torch operations
    else:
        raise ValueError(
            "Values of p different from 2 and Inf are currently not supported..."
        )
    return v


def universal_perturbation(
    dataset,
    net,
    grads,
    delta=0.2,
    max_iter_uni=np.inf,
    xi=10,
    p=np.inf,
    num_classes=10,
    overshoot=0.02,
    max_iter_df=10,
    device="cuda",  # Default to CUDA device
):
    """
    Generate universal perturbations for adversarial attacks.
    :param dataset: Images of size MxHxWxC (M: number of images)
    :param net: neural network model (input: images, output: activation values before softmax)
    :param grads: gradient functions (one for each class)
    :param delta: desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: maximum number of iterations for universal perturbation
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (only p=2 or p=inf are supported)
    :param num_classes: number of classes in the classification task
    :param overshoot: used to prevent vanishing updates
    :param max_iter_df: maximum number of iterations for deepfool
    :param device: device for computation ('cuda' or 'cpu')
    :return: the universal perturbation v
    """

    # Ensure dataset is a tensor for easier manipulation
    dataset = torch.tensor(dataset, dtype=torch.float32, device=device)

    # Initialize perturbation tensor with the shape of the first image
    v = torch.zeros_like(dataset[0], device=device)  # Initialize perturbation
    fooling_rate = 0.0
    num_images = dataset.shape[0]  # Number of images in the dataset
    itr = 0

    while fooling_rate < 1 - delta and itr < max_iter_uni:
        np.random.shuffle(dataset.cpu().numpy())  # Shuffle dataset for each iteration

        print(f"Starting pass number {itr}")

        # Go through the dataset and compute perturbation increments
        for k in range(num_images):
            cur_img = dataset[
                k : k + 1
            ]  # Select current image as tensor (batch size of 1)
            cur_img = cur_img.permute(0, 3, 1, 2)
            output_orig = net(cur_img)  # Get output for the original image
            print(cur_img.shape)
            print(v.shape)
            v = torch.tensor(v).permute(2, 0, 1).unsqueeze(0).to(device)
            output_perturbed = net(cur_img + v)  # Get output for the perturbed image

            # If the predicted class is the same for both the original and perturbed images
            if int(torch.argmax(output_orig).item()) == int(
                torch.argmax(output_perturbed).item()
            ):
                print(f">> k = {k}, pass #{itr}")

                # Compute adversarial perturbation using DeepFool
                dr, iter, _, _ = deepfool(
                    cur_img + v,
                    net,
                    num_classes=num_classes,
                    overshoot=overshoot,
                    max_iter=max_iter_df,
                )

                if iter < max_iter_df - 1:  # Ensure convergence
                    v = v + dr  # Update the perturbation
                    v = proj_lp(v, xi, p)  # Project the perturbation onto the lp ball

        itr += 1

        # Perturb the dataset with the computed perturbation
        dataset_perturbed = dataset + v  # Add perturbation to the entire dataset

        est_labels_orig = np.zeros(num_images, dtype=int)
        est_labels_pert = np.zeros(num_images, dtype=int)

        batch_size = 100
        num_batches = int(np.ceil(num_images / batch_size))  # Number of batches

        # Compute the estimated labels in batches
        for ii in range(num_batches):
            m = ii * batch_size
            M = min((ii + 1) * batch_size, num_images)
            est_labels_orig[m:M] = (
                torch.argmax(net(dataset[m:M]), dim=1).cpu().numpy()
            )  # Get predictions for original
            est_labels_pert[m:M] = (
                torch.argmax(net(dataset_perturbed[m:M]), dim=1).cpu().numpy()
            )  # Get predictions for perturbed

        # Compute the fooling rate
        fooling_rate = float(np.sum(est_labels_pert != est_labels_orig)) / num_images
        print(f"FOOLING RATE = {fooling_rate}")

    return v
