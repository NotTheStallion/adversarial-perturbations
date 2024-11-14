"""
This module contains the implementation of the DeepFool algorithm for generating adversarial perturbations.
"""

import numpy as np
import torch
import copy


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):
    """
    DeepFool algorithm for adversarial attacks.

    :param image: Image of size HxWx3
    :param net: Pre-trained neural network model (input: images, output: activation values BEFORE softmax).
    :param num_classes: Number of classes to test against. Limits the number of outputs considered. Default = 10.
    :param overshoot: Overshoot factor to prevent vanishing updates. Default = 0.02.
    :param max_iter: Maximum number of iterations. Default = 50.
    :return: Perturbation that fools the classifier, number of iterations, original label, new estimated label, and perturbed image.
    """

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        image = image.cuda()
        net = net.cuda()

    # Getting probability vector
    f_image = (
        net.forward(image.unsqueeze(0).requires_grad_(True))
        .detach()
        .cpu()
        .numpy()
        .flatten()
    )
    # Getting top num_classes predictions
    I = np.argsort(f_image)[::-1][:num_classes]

    # Label of the original image
    label_orig = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    # Perturbation vector
    w = np.zeros(input_shape)
    # Accumulated perturbation
    r_tot = np.zeros(input_shape)

    iter = 0

    x = pert_image.unsqueeze(0).requires_grad_(
        True
    )  # Add batch dimension and enable gradient calculation
    # print(f"Input shape: {x.shape}")

    # Prediction of perturbed image
    pred_p = net.forward(x)
    # print(f"Prediction: {pred_p[10]}")
    label_pert = label_orig

    while label_pert == label_orig and iter < max_iter:
        pert = np.inf

        pred_p[0, label_orig].backward(
            retain_graph=True
        )  # Compute gradients for the original class
        grad_origin = (
            x.grad.detach().cpu().numpy().copy()
        )  # Store the original class gradient

        for k in range(1, num_classes):
            x.grad.zero_()

            pred_p[0, I[k]].backward(
                retain_graph=True
            )  # Backpropagate to get gradient of class `I[k]`
            cur_grad = (
                x.grad.detach().cpu().numpy().copy()
            )  # Store the gradient of the current class

            # w_k is the direction to move in order to change class
            w_k = cur_grad - grad_origin  # Eq 8 in the paper

            # Difference in activation between current class and original class
            f_k = (pred_p[0, I[k]] - pred_p[0, label_orig]).item()  # Eq 8 in the paper

            # Formula: perturbation = |f_k| / ||w_k|| (L2 norm)
            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())  # Eq 8 in the paper

            # Update the perturbation if a smaller one is found
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # Update the total perturbation
        r_i = pert * w / np.linalg.norm(w)  # Eq : 9
        r_tot = r_tot + r_i  # Accumulate total perturbation

        # Apply the perturbation to the image
        pert_image = image + torch.from_numpy(r_tot).to(image.device)

        # Perform forward pass on the perturbed image
        x = pert_image.requires_grad_(
            True
        )  # Recreate the perturbed image tensor with gradient tracking
        input = x.view(x.size()[-4:]).type(
            torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
        )  # Flatten the input
        pred_p = net.forward(input)  # Forward pass through the network
        label_pert = np.argmax(
            pred_p.detach().cpu().numpy().flatten()
        )  # Predicted class for the perturbed image

        iter += 1  # Increment the iteration counter

    # Final perturbation scaling by (1 + overshoot)
    r_tot = r_tot  # Scaling the perturbation vector

    return (
        r_tot,
        iter,
        label_orig,
        label_pert,
        pert_image,
    )  # Return the total perturbation, iteration count, original and new labels, and perturbed image




def local_deepfool(
    image, net, num_classes=10, overshoot=0.02, max_iter=50, region_mask=None, verbose=True
):
    """
    region_mask : A matrix that contains 1 where you want the perturbation to be applied and 0 elsewhere. By default the perturbation is applied to the entire image.
    """
    
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        image = image.cuda()
        net = net.cuda()

    input_shape = image.cpu().numpy().shape

    f_image = (
        net.forward(image.unsqueeze(0).requires_grad_(True))
        .detach()
        .cpu()
        .numpy()
        .flatten()
    )

    I = np.argsort(f_image)[::-1][:num_classes]
    label_orig = I[0]

    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    iter = 0
    x = pert_image.unsqueeze(0).requires_grad_(True)
    pred_p = net.forward(x)
    label_pert = label_orig

    while label_pert == label_orig and iter < max_iter:
        pert = np.inf

        pred_p[0, label_orig].backward(retain_graph=True)
        grad_origin = x.grad.detach().cpu().numpy().copy()

        for k in range(1, num_classes):
            x.grad.zero_()

            pred_p[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.detach().cpu().numpy().copy()

            w_k = np.zeros_like(cur_grad)

            if region_mask is None:
                w_k[0] = cur_grad[0] - grad_origin[0]
            else:
                w_k[0] = (cur_grad[0] - grad_origin[0]) * region_mask

            f_k = (pred_p[0, I[k]] - pred_p[0, label_orig]).item()  # Eq 8 in the paper
            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())  # Eq 8 in the paper

            if pert_k < pert:
                pert = pert_k
                w = w_k

        r_i = pert * w / np.linalg.norm(w)  # Eq : 9
        if region_mask is None:
            r_tot += r_i[0]
        else:
            r_tot += r_i[0] * region_mask

        pert_image = image + torch.from_numpy(r_tot).to(image.device)

        x = pert_image.unsqueeze(0).requires_grad_(True)
        input = x.view(x.size()[-4:]).type(
            torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
        )

        pred_p = net.forward(input)
        label_pert = np.argmax(pred_p.detach().cpu().numpy().flatten())

        iter += 1
        if iter % 100 == 0 and verbose:
            print(f"Iteration: {iter}, label_pert: {label_pert}/{label_orig}")

    # Scale final perturbation
    r_tot = r_tot
    if verbose:
        print(f"original label: {label_orig}, perturbed label: {label_pert}")
    return r_tot, iter, label_orig, label_pert, pert_image

