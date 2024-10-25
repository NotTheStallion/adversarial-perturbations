import os
import torch
import random
import trainer as trainer_module
import data_loader
import matplotlib.pyplot as plt
import adversarial_perturbation
import numpy as np

# Store model path
MODEL_PATH = "fashion_mnist_model.pth"


def main():
    # Define the device for computation (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    trainset, testset = data_loader.load_data()
    trainer = trainer_module.trainer()

    # Move the model to the selected device
    trainer.net = trainer.net.to(device)

    # Check if a pre-trained model exists and load it
    if os.path.exists(MODEL_PATH):
        print("Loading pre-trained model...")
        trainer.net.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True)
        )
        trainer.net.eval()
        accuracy = trainer.evaluate(testset)
    else:
        print("No pre-trained model found. Training the model...")
        accuracy = trainer.train(trainset, testset)
        # Save the model after training
        torch.save(trainer.net.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    # Generate adversarial perturbations
    v, fooling_rates, accuracies, total_iterations = adversarial_perturbation.generate(
        accuracy, trainset, testset, trainer.net, delta=0.2
    )

    # Plot the fooling rates
    plt.title("Fooling Rates over Universal Iterations")
    plt.xlabel("Universal Algorithm Iter")
    plt.ylabel("Fooling Rate on test data")
    plt.plot(total_iterations, fooling_rates)
    plt.show()

    # Normalize and visualize the perturbation v
    if isinstance(v, np.ndarray):
        v_min = v.min()
        v_max = v.max()
        v_normalized = (v - v_min) / (v_max - v_min)
    else:
        v_min = v.min()
        v_max = v.max()
        v_normalized = (v - v_min) / (v_max - v_min)
        v_normalized = v_normalized.detach().cpu().numpy()

    v_normalized = v_normalized.squeeze()  # This will change the shape to (28, 28)

    plt.title("Adversarial Perturbation v")
    plt.imshow(v_normalized, cmap="gray")
    plt.colorbar()
    plt.axis("off")
    plt.show()

    # Display 5 random images with their perturbed versions
    dataiter = iter(testset)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    indices = random.sample(range(len(images)), 5)
    images = images[indices]
    labels = labels[indices]

    # Assuming 'v' is initialized as a numpy array or tensor representing perturbation
    v = torch.tensor(v, dtype=torch.float32, device=device)

    images = images.to(v.device)  # Move images to the same device as v if necessary

    # Now apply v to images
    # Since 'v' is already of shape [1, 1, 28, 28], you can expand it across the batch dimension
    perturbed_images = images + v.expand(images.size(0), -1, -1, -1)

    # Get predictions for perturbed images
    with torch.no_grad():
        outputs = trainer.net(perturbed_images)
        _, predicted_labels = torch.max(outputs, 1)

    # Plot the original and perturbed images
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    images_np = images.detach().cpu().numpy()

    for i in range(5):
        axs[0, i].imshow(images_np[i].squeeze(), cmap="gray")
        axs[0, i].set_title(f"Label: {labels[i].item()}")
        axs[0, i].axis("off")

        perturbed_image_np = perturbed_images[i].detach().cpu().numpy()
        axs[1, i].imshow(perturbed_image_np.squeeze(), cmap="gray")
        axs[1, i].set_title(f"Pred: {predicted_labels[i].item()}")
        axs[1, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
