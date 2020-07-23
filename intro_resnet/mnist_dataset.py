import torch
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST
from torchvision import transforms

def mnist_data():
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

    train = MNIST(root='../mnist/', train=True, transform=transform)
    x_train = train.data/255.
    y_train = train.targets

    test = MNIST(root='../mnist/', transform=transform)
    x_test = test.data/255.
    y_test = test.targets    

    return x_train, y_train, x_test, y_test

def create_TensorDataset(x_train, y_train, x_test, y_test):
    train = torch.utils.data.TensorDataset(x_train, y_train)
    test = torch.utils.data.TensorDataset(x_test, y_test)

    return train, test

def create_DataLoader(train=None, test=None, batch_size=100):
    if train is None and test is None:
        x_train, y_train, x_test, y_test = mnist_data()
        train, test = create_TensorDataset(x_train, y_train, x_test, y_test)

    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader    