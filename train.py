import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer with input channels=3 (RGB image), output channels=6, kernel size=5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling layer with kernel size=2 and stride=2
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional layer with input channels=6, output channels=16, kernel size=5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layer with input features=16*5*5 (from conv2 layer), output features=120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Second fully connected layer with input features=120, output features=84
        self.fc2 = nn.Linear(120, 84)
        # Third fully connected layer with input features=84, output features=10 (number of classes)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Apply first conv layer followed by pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second conv layer followed by pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 16 * 5 * 5)
        # Apply first fully connected layer with relu activation
        x = F.relu(self.fc1(x))
        # Apply second fully connected layer with relu activation
        x = F.relu(self.fc2(x))
        # Apply third fully connected layer with no activation (raw scores)
        x = self.fc3(x)
        return x


def main():
    # Set the device to GPU if available, else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Transformations for the input data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    # Create an instance of the Net class
    net = Net()
    net.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training the network
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Optimize
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # Print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
                running_loss = 0.0

    print('Finished Training')

    # Save the trained model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

if __name__ == '__main__':
    main()