import model
import torch
from torch import nn
from torch import optim

# Define the device for training and evaluation (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class trainer:
    def __init__(self):
        # Initialize the model and transfer it to the chosen device
        self.net = model.ConvNet()
        self.net.to(device)

        # Define the loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        self.n_epochs = 4

    def train(self, trainloader, testloader):
        accuracy = 0
        self.net.train()  # Set the model to training mode
        for epoch in range(self.n_epochs):
            running_loss = 0.0
            print_every = 200  # Print loss every 200 mini-batches

            for i, (inputs, labels) in enumerate(trainloader, 0):
                # Transfer inputs and labels to the device
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass, backward pass, and optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Accumulate the running loss
                running_loss += loss.item()
                if (i % print_every) == (print_every - 1):
                    print(f"[{epoch+1}, {i+1}] loss: {running_loss / print_every:.3f}")
                    running_loss = 0.0

            # Print accuracy after every epoch
            accuracy = self.evaluate(testloader)
            print(
                f"Accuracy of the network on the test images after epoch {epoch+1}: {100 * accuracy:.2f} %"
            )

        print("Finished Training")
        return accuracy

    def evaluate(self, testloader):
        self.net.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        # Disable gradient calculation for evaluation
        with torch.no_grad():
            for images, labels in testloader:
                # Transfer images and labels to the device
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)

                # Calculate total and correct predictions
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate accuracy
        return correct / total
