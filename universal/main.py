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
    trainset, testset = data_loader.load_data()
    trainer = trainer_module.trainer()
    
    if os.path.exists(MODEL_PATH):
        print("Loading pre-trained model...")
        trainer.net.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
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

    # Print fooling rates over universal iterations
    plt.title("Fooling Rates over Universal Iterations")
    plt.xlabel("Universal Algorithm Iter")
    plt.ylabel("Fooling Rate on test data")
    plt.plot(total_iterations, fooling_rates)
    plt.show()

    # Print the perturbation v
    # If v is a numpy array, then normalize it between 0 and 1
    if isinstance(v, np.ndarray):
        v_min = v.min()
        v_max = v.max()
        v_normalized = (v - v_min) / (v_max - v_min)
    else:
        v_min = v.min()
        v_max = v.max()
        v_normalized = (v - v_min) / (v_max - v_min)
        v_normalized = v_normalized.detach().cpu().numpy()

    plt.title("Adversarial Perturbation v")
    plt.imshow(v_normalized, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.show()

    # Print 5 random images with their perturbed versions
    dataiter = iter(testset)
    images, labels = next(dataiter)
    indices = random.sample(range(len(images)), 5)

    images = images[indices]
    labels = labels[indices]
    images_np = images.detach().cpu().numpy()

    v = torch.tensor(v, device=images.device)
    v = v.permute(2, 0, 1)

    # Apply v
    perturbed_images = images + v.unsqueeze(0).expand_as(images)
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    perturbed_images = perturbed_images.float()

    # Get the new labels
    with torch.no_grad():
        outputs = trainer.net(perturbed_images)
        _, predicted_labels = torch.max(outputs, 1)

    # Plot
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(5):
        axs[0, i].imshow(images_np[i].squeeze(), cmap='gray')
        axs[0, i].set_title(f"Label: {labels[i].item()}")
        axs[0, i].axis('off')

        perturbed_image_np = perturbed_images[i].detach().cpu().numpy()
        axs[1, i].imshow(perturbed_image_np.squeeze(), cmap='gray')
        axs[1, i].set_title(f"Pred: {predicted_labels[i].item()}")
        axs[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()