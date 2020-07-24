from torchvision import models
from torchvision.models.resnet import ResNet, Bottleneck
import torch.nn as nn

class MnistResNet(ResNet):
    def __init__(self, block,layers,num_classes, pretrained=False, bias=False):
        super(MnistResNet, self).__init__(block = Bottleneck, layers=[3, 4, 6, 3])
        self.conv1 = nn.Conv2d(1, 
                               64, 
                               kernel_size = 3, 
                               stride=2, 
                               padding=3, 
                               bias=False)
        
def resnet50(num_classes=10):
    model = MnistResNet(block=models.resnet.Bottleneck, 
                        layers=[3, 4, 6, 3], 
                        num_classes = 10, 
                        pretrained=False, 
                        bias=False)

    return model

if  __name__ == "__main__":
    import resnet
    import mnist_dataset as  mnist
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = mnist.create_DataLoader() 
    model = MnistResNet() 
    resnet.train_model(model, train_loader, device)          
