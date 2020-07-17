import torch
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST

def mnist_data():
    mnist_data = MNIST(root='../mnist/')
    x_data = mnist_data.data/255.
    y_data = mnist_data.targets

    return x_data, y_data

def divide_train_test(x_data, y_data, test_size):
    features_train, features_test, targets_train, targets_test = train_test_split(x_data, y_data, test_size = test_size, random_state = 42)
    return features_train, features_test, targets_train, targets_test

def create_TensorDataset(features_train, targets_train, features_test, targets_test):
    train = torch.utils.data.TensorDataset(features_train, targets_train)
    test = torch.utils.data.TensorDataset(features_test, targets_test)

    return train, test

def create_DataLoader(train, test, batch_size):
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader    
