import torch
from torch import nn
import torch.optim as optim
import numpy as np

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_model(model, PATH):
    torch.save(model.state_dict(), PATH)

def train_model(model, train_loader, num_epochs=80, learning_rate=0.001, device='cpu', update_lr=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f"Epoch {epoch + 1} / {num_epochs}")

        model.train()
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # print statistics
            if (i + 1) % 1000 == 0:    
                print (f'Step [{i + 1}/{len(train_loader)}] Loss: {loss.item()}')

        if update_lr is True:
            if (epoch+1) % 20 == 0:
                learning_rate /= 3
                update_lr(optimizer, learning_rate)

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)

    print('Finished Training')

    return losses

def test_model(model, test_loader, batch_size, num_classes, classes, device='cpu'):
    accuracy = 0
    class_total = list(0. for i in range(num_classes))
    class_correct = list(0. for i in range(num_classes))

    model.eval()
    with torch.no_grad():
        total = 0
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs01 = model(images)
            _, predicted = torch.max(outputs01, 1)
            c = (predicted == labels).squeeze()
            total01 += labels.size(0)
            accuracy01 += (predicted == labels).sum().item()

            for i in range(batch_size):
                label = labels[i]

                class_correct[label] += c[i].item()
                class_total[label] += 1

    if classes:
        for i in range(num_classes):
            print(f"Accuracy of {classes[i]:10s} : {class_correct[i]/class_total[i] * 100:2d}%")
    print(f"Total accuracy: {accuracy / total * 100:.2f}%")

    if classes:
        return accuracy / total * 100, np.array(class_correct/class_total * 100)
    else:
        return accuracy / total * 100
