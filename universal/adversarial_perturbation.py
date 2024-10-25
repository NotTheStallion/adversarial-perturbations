import numpy as np
import deepfool
from PIL import Image
import torch
from torchvision import transforms


def project_perturbation(data_point, p, perturbation):
    if p == 2:
        perturbation = perturbation * min(
            1, data_point / np.linalg.norm(perturbation.flatten(1))
        )
    elif p == np.inf:
        perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), data_point)
    return perturbation


def generate(
    accuracy,
    trainset,
    testset,
    net,
    delta=0.2,
    max_iter_uni=np.inf,
    xi=10,
    p=np.inf,
    num_classes=10,
    overshoot=0.2,
    max_iter_df=20,
):
    """
    Generates a universal adversarial perturbation for a neural network.
    """
    # Set the network to evaluation mode
    net.eval()
    device = next(net.parameters()).device

    # Importing images and creating an array with them
    img_trn = []
    for image in trainset:
        for image2 in image[0]:
            img_trn.append(image2.numpy())

    img_tst = []
    for image in testset:
        for image2 in image[0]:
            img_tst.append(image2.numpy())

    # Setting the number of images to 100 for training
    num_img_trn = 100
    index_order = np.arange(num_img_trn)

    # Initializing the perturbation to zeros as a 3D tensor
    v = np.zeros((1, 1, 28, 28), dtype=np.float32)  # Shape (1, 1, 28, 28)

    # Initializing fooling rate and iteration count
    fooling_rate = 0.0
    iter = 0

    # Transformers to be applied to images in order to feed them to the network
    transformer1 = transforms.Compose([transforms.ToTensor()])
    transformer2 = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(28)]
    )

    fooling_rates = [0]
    accuracies = [accuracy]
    total_iterations = [0]

    # Begin of the main loop on Universal Adversarial Perturbations algorithm
    while fooling_rate < 1 - delta and iter < max_iter_uni:
        np.random.shuffle(index_order)
        print("Iteration  ", iter)

        for index in index_order:
            # Generating the original image from data
            cur_img = Image.fromarray(img_trn[index][0])
            cur_img1 = transformer1(transformer2(cur_img))[np.newaxis, :].to(device)

            # Feeding the original image to the network and storing the label returned
            r2 = net(cur_img1).max(1)[1]
            torch.cuda.empty_cache()

            # Generating a perturbed image from the current perturbation v and the original image
            per_img_array = (
                transformer2(cur_img) + v.squeeze()
            )  # Use v.squeeze() directly
            per_img = Image.fromarray(per_img_array.astype(np.uint8))
            per_img1 = transformer1(per_img)[np.newaxis, :].to(device)

            # Feeding the perturbed image to the network and storing the label returned
            r1 = net(per_img1).max(1)[1]
            torch.cuda.empty_cache()

            # If the label of both images is the same, update the perturbation v
            if r1 == r2:
                """ print(
                    ">> k =",
                    np.where(index == index_order)[0][0],
                    ", pass #",
                    iter,
                    end="      ",
                ) """

                # Finding a new minimal perturbation with deepfool to fool the network on this image
                dr, iter_k, label, k_i, pert_image = deepfool.deepfool(
                    per_img1[0],
                    net,
                    num_classes=num_classes,
                    overshoot=overshoot,
                    max_iter=max_iter_df,
                )

                # Adding the new perturbation found and projecting the perturbation v and data point xi on p.
                if iter_k < max_iter_df - 1:
                    # Ensure to maintain the shape of v
                    v += dr[0, 0, :, :][None, None, :, :]  # Add dimensions to dr
                    v = project_perturbation(xi, p, v)

        iter += 1

        # v_tensor is created as a 3D tensor already
        v_tensor = torch.tensor(v, dtype=torch.float32).to(
            device
        )  # Shape (1, 1, 28, 28)

        with torch.no_grad():
            # Compute fooling_rate
            labels_original_images = []
            labels_perturbed_images = []
            i = 0

            # Finding labels for original images
            for inputs, _ in testset:
                i += inputs.size(0)
                inputs = inputs.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                labels_original_images.append(
                    predicted
                )  # Collect predictions directly as tensors

            # Concatenate all predictions for original images
            labels_original_images = torch.cat(labels_original_images).to(
                device
            )  # Ensure on the same device

            correct = 0

            # Finding labels for perturbed images
            for inputs, labels in testset:
                inputs = inputs.to(device)
                perturbed_inputs = (
                    inputs + v_tensor.float()
                )  # Use the tensor for the perturbation
                outputs = net(perturbed_inputs)
                _, predicted = outputs.max(1)
                labels_perturbed_images.append(
                    predicted
                )  # Collect predictions directly as tensors
                correct += (predicted == labels.to(device)).sum().item()

            # Concatenate all predictions for perturbed images
            labels_perturbed_images = torch.cat(labels_perturbed_images).to(
                device
            )  # Ensure on the same device

            # Calculating the fooling rate by dividing the number of fooled images by the total number of images
            fooling_rate = float(
                torch.sum(labels_original_images != labels_perturbed_images)
            ) / float(i)

            print()
            print("FOOLING RATE: ", fooling_rate)
            fooling_rates.append(fooling_rate)
            accuracies.append(correct / i)
            total_iterations.append(iter)

    return v, fooling_rates, accuracies, total_iterations
