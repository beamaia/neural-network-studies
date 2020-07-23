from torchvision.models.resnet import ResNet, BasicBlock
import torch

class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes = 10)
        self.conv1 = torch.nn.Conv2d(1, 64, 
                                     kernel_size = (7,7), 
                                     stride=(2,2), 
                                     padding=(3,3), bias=False)

if  __name__ == "__main__":
    import resnet
    import dataset_mnist as  mnist
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = mnist.create_DataLoader() 
    model = MnistResNet() 
    accuracy_scores = resnet.train_model(model, train_loader, device)          
