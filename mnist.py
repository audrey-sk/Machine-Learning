import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

train= datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainload = DataLoader(train, batch_size=64, shuffle=True)
testload = DataLoader(test, batch_size=64, shuffle=True )

class NeuralNet(nn.Module):
  def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = NeuralNet()
criterion= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(), lr=0.001)

def calculate_accuracy(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


num_epochs = 10
# Initialize accuracy lists
train_acc = []  #store training accuracy for each epoch
val_acc = []    #store validation accuracy for each epoch
losses=[]
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, labels in trainload:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(trainload)
    losses.append(loss.item())

    model.eval()
    train_accuracy = calculate_accuracy(trainload)
    validation_accuracy = calculate_accuracy(testload)
    train_acc.append(train_accuracy)
    val_acc.append(validation_accuracy)



    print(f"Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {validation_accuracy:.4f} Loss:{loss:.4f}")

plt.plot(train_acc, label='Training Accuracy', color='red')
plt.plot(val_acc, label='Validation Accuracy', color='blue')
plt.plot(losses, label='Loss', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()