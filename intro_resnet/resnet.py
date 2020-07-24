from sklearn.metrics import accuracy_score
import torch
from torch import nn
import numpy as np

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_model(model, data_loader, device):
    model.train()
    num_epochs = int(input("Number of epochs: "))
    learning_rate = float(input("Learning rate: "))

    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(epoch)
        print(f'Epoch {epoch + 1} of {num_epochs}.')
        count = 0
 
        for images, labels in data_loader:
            images.resize_([100,3,28,28]).to(device)
            labels.to(device)

            # Forward pass
            outputs = model(images)
            # outputs = torch.max(outputs.data, 1)[1].cpu().numpy()

            loss = error(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (count + 1) % 100 == 0:
                print (f'Step [{count + 1}/{len(data_loader)}] Loss: {loss.item()}')

            count += 1
        
        # change_learning_rate = int(input("Change learning rate?\n0: No   1: Yes"))

        if epoch % 20 == 0:
            learning_rate /= 3
            update_lr(optimizer, learning_rate)

def test_model(model, data_loader, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loader:
            images = images.resize_(100, 1, 32, 32).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print('Accuracy of the model on the test images: {.2f} %'.format(accuracy))

      
    return accuracy
