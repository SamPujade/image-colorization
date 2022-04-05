import torch.nn as nn
import torch

class Generator(nn.Module):

    def get_layers(self, ch_in, ch_out, kernel_size=4, stride=2, padding=1, norm=True, act=True, leaky=True, transpose=False, dropout=False):
        """
        Construct a convolutional unit with a conv layer
        followed by a batch normalisation layer and Leaky ReLU.
        """
        layers = []
        if transpose:
            layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel_size, stride, padding))
        else:
            layers.append(nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, bias=False))
        if norm:
            layers.append(nn.BatchNorm2d(ch_out))
        if act:
            if leaky:
                layers.append(nn.LeakyReLU(0.2, True))
            else:
                layers.append(nn.ReLU(True))
        if dropout:
            layers.append(nn.Dropout(0.5))

        return nn.Sequential(*layers)
    
    def __init__(self):
        """
        In the constructer, all the convolutional, upsampling and max pooling 
        units are instantiated and assigned as member variables. 
        """
        super().__init__()
        
        # Encoding layers
        self.conv_en1 = self.get_layers(1, 64, kernel_size=1, stride=1, padding=0, norm=False, act=False)
        self.conv_en2 = self.get_layers(64, 128)
        self.conv_en3 = self.get_layers(128, 256)
        self.conv_en4 = self.get_layers(256, 512)
        self.conv_en5 = self.get_layers(512, 512)
        self.conv_en6 = self.get_layers(512, 512)
        self.conv_en7 = self.get_layers(512, 512)
        self.conv_en8 = self.get_layers(512, 512, norm=False)
        
        # Decoding layers
        self.conv_de1 = self.get_layers(512, 512, transpose=True, leaky=False, dropout=True)
        self.conv_de2 = self.get_layers(1024, 512, transpose=True, leaky=False, dropout=True)
        self.conv_de3 = self.get_layers(1024, 512, transpose=True, leaky=False, dropout=True)
        self.conv_de4 = self.get_layers(1024, 512, transpose=True, leaky=False, dropout=True)
        self.conv_de5 = self.get_layers(1024, 256, transpose=True, leaky=False)
        self.conv_de6 = self.get_layers(512, 128, transpose=True, leaky=False)
        self.conv_de7 = self.get_layers(256, 64, transpose=True, leaky=False)
        self.conv_de8 = self.get_layers(128, 2, stride=1, kernel_size=1, padding=0, norm=False, act=False, transpose=True, leaky=False)

    def forward(self, x):
        """ 
        An input tensor of a black and white image is accepted and
        passed through the U-Net model. A colored image in CieLAB color
        space is returned as the result. 
        """
        # Encoding path
        x1 = self.conv_en1(x)   # [batch, 1, 256, 256] => [batch, 64, 256, 256]
        x2 = self.conv_en2(x1)   # [batch, 64, 256, 256] => [batch, 128, 128, 128]
        x3 = self.conv_en3(x2)   # [batch, 128, 128, 128] => [batch, 256, 64, 64]
        x4 = self.conv_en4(x3)   # [batch, 256, 64, 64] => [batch, 512, 32, 32]
        x5 = self.conv_en5(x4)   # [batch, 512, 32, 32] => [batch, 512, 16, 16]
        x6 = self.conv_en6(x5)   # [batch, 512, 16, 16] => [batch, 512, 8, 8]
        x7 = self.conv_en7(x6)   # [batch, 512, 8, 8] => [batch, 512, 4, 4]
        x8 = self.conv_en8(x7)   # [batch, 512, 4, 4] => [batch, 512, 2, 2]
        
        # Decoding path
        x_out = torch.cat([x7, self.conv_de1(x8)], 1)   # [batch, 512, 2, 2] => [batch, 2*512, 4, 4]
        x_out = torch.cat([x6, self.conv_de2(x_out)], 1)   # [batch, 2*512, 4, 4] => [batch, 2*512, 8, 8]
        x_out = torch.cat([x5, self.conv_de3(x_out)], 1)   # [batch, 2*512, 8, 8 => [batch, 2*512, 16, 16]
        x_out = torch.cat([x4, self.conv_de4(x_out)], 1)   # [batch, 2*512, 16, 16] => [batch, 2*512, 32, 32]
        x_out = torch.cat([x3, self.conv_de5(x_out)], 1)   # [batch, 2*512, 32, 32] => [batch, 2*256, 64, 64]
        x_out = torch.cat([x2, self.conv_de6(x_out)], 1)   # [batch, 2*256, 64, 64] => [batch, 2*128, 128, 128]
        x_out = torch.cat([x1, self.conv_de7(x_out)], 1)   # [batch, 2*128, 128, 128] => [batch, 2*64, 256, 256]
        x_out = self.conv_de8(x_out)   # [batch, 2*64, 256, 256] => [batch, 2, 256, 256]
        x_out = nn.Tanh()(x_out)
        
        return x_out