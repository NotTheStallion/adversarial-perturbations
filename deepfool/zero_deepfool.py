import numpy as np
import torch
import copy

def zero_deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):
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
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")


    # Getting probability vector
    f_image = net.forward(image.unsqueeze(0).requires_grad_(True)).detach().cpu().numpy().flatten()
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
    
    x = pert_image.unsqueeze(0).requires_grad_(True)  # Add batch dimension and enable gradient calculation
    # print(f"Input shape: {x.shape}")
    
    # Prediction of perturbed image
    pred_p = net.forward(x)
    # print(f"Prediction: {pred_p[10]}")
    label_pert = label_orig


    while label_pert == label_orig and iter < max_iter:
        pert = np.inf
        
        pred_p[0, label_orig].backward(retain_graph=True)  # Compute gradients for the original class
        grad_origin = x.grad.detach().cpu().numpy().copy()  # Store the original class gradient

        for k in range(1, num_classes):
            x.grad.zero_()

            pred_p[0, I[k]].backward(retain_graph=True)  # Backpropagate to get gradient of class `I[k]`
            cur_grad = x.grad.detach().cpu().numpy().copy()  # Store the gradient of the current class

            # w_k is the direction to move in order to change class
            w_k = cur_grad - grad_origin  # Eq 8 in the paper
            
            # Difference in activation between current class and original class
            f_k = (pred_p[0, I[k]] - pred_p[0, label_orig]).item() # Eq 8 in the paper

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
        x = pert_image.requires_grad_(True)  # Recreate the perturbed image tensor with gradient tracking
        input = x.view(x.size()[-4:]).type(torch.cuda.FloatTensor if is_cuda else torch.FloatTensor)  # Flatten the input
        pred_p = net.forward(input)  # Forward pass through the network
        label_pert = np.argmax(pred_p.detach().cpu().numpy().flatten())  # Predicted class for the perturbed image

        iter += 1  # Increment the iteration counter

    # Final perturbation scaling by (1 + overshoot)
    r_tot = r_tot  # Scaling the perturbation vector
    
    return r_tot, iter, label_orig, label_pert, pert_image  # Return the total perturbation, iteration count, original and new labels, and perturbed image



if __name__ == "__main__":
    import torch.nn as nn
    import torchvision.transforms as transforms
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torchvision.models as models
    from PIL import Image
    from zero_deepfool import zero_deepfool  # Assuming deepfool.py is defined as before
    import os
    from torchvision.models import ResNet34_Weights

    def test_one_image():
        # Load pretrained ResNet-34 model
        net = models.resnet34(weights=ResNet34_Weights.DEFAULT)

        # Switch to evaluation mode
        net.eval()

        # Load the image
        im_orig = Image.open(f"data/test_img1.jpg")  # Change the path as needed for your test image

        # Mean and std used for normalization (ImageNet stats)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Preprocessing the image: resize, crop, convert to tensor, and normalize
        im = transforms.Compose(
            [
                transforms.Resize(256),  # Rescale image to 256x256
                transforms.CenterCrop(224),  # Crop to 224x224
                transforms.ToTensor(),  # Convert to PyTorch Tensor
                transforms.Normalize(mean=mean, std=std),  # Normalize with ImageNet mean and std
            ]
        )(im_orig)

        # Run DeepFool attack
        r, loop_i, label_orig, label_pert, pert_image = zero_deepfool(im, net)

        # Load class labels from file (assuming ImageNet labels are in synset_words.txt)
        labels = open(os.path.join("data/synset_words.txt"), "r").read().split("\n")

        # Get original and perturbed class labels
        str_label_orig = labels[int(label_orig)].split(",")[0]  # Original label
        str_label_pert = labels[int(label_pert)].split(",")[0]  # Perturbed label

        return str_label_orig, str_label_pert


    # Test a single image
    original_label, perturbed_label = test_one_image()

    print(f"Original label: {original_label} -> Perturbed label: {perturbed_label}")