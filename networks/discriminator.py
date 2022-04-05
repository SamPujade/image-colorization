import torch.nn as nn
import torch


class Discriminator(nn.Module):
        
    def get_layers(self, ch_in, ch_out, kernel_size=4, stride=2, padding=1, norm=True, act=True):
        """
        Construct a convolutional unit with a conv layer
        followed by a batch normalisation layer and Leaky ReLU.
        """
        layers = [nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding)]
        if norm:
            layers.append(nn.BatchNorm2d(ch_out))
        if act:
            layers.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*layers)
    
    def __init__(self):
        super().__init__()
        """
        In the constructer, all the convolutional and max pooling units 
        are instantiated and assigned as member variables. 
        """
        # First layer
        self.conv1 = self.get_layers(3, 64, norm=False)
        # Second layer
        self.conv2 = self.get_layers(64, 128)
        # Third layer
        self.conv3 = self.get_layers(128, 256)
        # Fourth layer
        self.conv4 = self.get_layers(256, 512, stride=1)
        # Fifth layer
        self.conv5 = self.get_layers(512, 1, stride=1, norm=False, act=False)

        self.linear = nn.Linear(900, 1)

    def forward(self, x):
        """ 
        An input tensor of a colored image from either the generator or source
        is accepted and passed through the model. The probability of the image
        belonging to the source domain is returned as the result. 
        """
        x1 = self.conv1(x)   # [batch, 3, 256, 256] => [batch, 64, 128, 128]
        x2 = self.conv2(x1)   # [batch, 64, 128, 128] => [batch, 128, 64, 64]
        x3 = self.conv3(x2)   # [batch, 128, 64, 64] => [batch, 256, 32, 32]
        x4 = self.conv4(x3)   # [batch, 256, 32, 32] => [batch, 512, 31, 31]
        x5 = self.conv5(x4)   # [batch, 512, 31, 31] => [batch, 1, 30, 30]
        x_last = self.linear(torch.flatten(x5, start_dim=1))   # [batch, 1, 30, 30] => [batch, 1]
        x_last = nn.Sigmoid()(x_last.squeeze())
        
        return x_last