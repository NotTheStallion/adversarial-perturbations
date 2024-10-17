import numpy as np
import torch
import copy

def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):
    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    # Forward pass to get the output of the network
    f_image = net.forward(image.unsqueeze(0).requires_grad_(True)).detach().cpu().numpy().flatten()
    I = np.argsort(f_image)[::-1][:num_classes]  # Sort and take the top num_classes

    label = I[0]  # Get original label

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = pert_image.unsqueeze(0).requires_grad_(True)
    print(f"Input shape: {x.shape}")
    fs = net.forward(x)
    k_i = label

    while k_i == label and loop_i < max_iter:
        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            x.grad.zero_()  # Clear gradients

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # Set new w_k and f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).item()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # Determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # Compute r_i and r_tot (with small constant for numerical stability)
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = r_tot + r_i

        # Apply perturbation to the image
        pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).to(image.device)

        # Forward pass with the perturbed image
        x = pert_image.unsqueeze(0).requires_grad_(True)
        print(f"Input shape: {x.shape}")
        input = x.view(x.size()[-4:]).type(torch.cuda.FloatTensor if is_cuda else torch.FloatTensor)
        fs = net.forward(input)
        k_i = np.argmax(fs.detach().cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image
