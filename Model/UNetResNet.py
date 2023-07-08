import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

class RenseNetEncoderBlock(nn.Module):
    """Some Information about EncoderBlock"""

    def __init__(self, backbone):
        super(RenseNetEncoderBlock, self).__init__()
        if backbone.lower() == 'resnet18':
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            
        elif backbone.lower() == 'resnet34':
            self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            
        elif backbone.lower() == 'resnet50':
            self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            
        elif backbone.lower() == 'resnet101':
            self.resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        
        elif backbone.lower() == 'resnet152':
            self.resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)

    def forward(self, x):
        features = [x]
        modules = list(self.resnet.children())
        encoder = torch.nn.Sequential(*(list(modules)[:-2]))
        # i = 1
        for layer in encoder:
            features.append(layer(features[-1]))

        return features

class DecoderBLock(nn.Module):
    """Some Information about DecoderBLock"""

    def __init__(self, skip_channels, x_channels, kernel_size):
        super(DecoderBLock, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=skip_channels + x_channels,
                out_channels=skip_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                bias=True), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=skip_channels,
                out_channels=skip_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                bias=True), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        

    def forward(self, skip, x):
        x = F.interpolate(x, size=[skip.shape[2], skip.shape[3]], mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1)
        x = self.decoder(x)
        
        return x

class UNetResNet(nn.Module):
    """Some Information about UNetResNet"""

    def __init__(self, device, backbone):
        super(UNetResNet, self).__init__()
        self.backbone = backbone.lower()
        self.encoder = RenseNetEncoderBlock(backbone).to(device)
        
        # features = size of last channel
        if backbone.lower() == 'resnet18':
            features = [64, 64, 128, 256, 512]
        elif backbone.lower() == 'resnet34':
            features = [64, 64, 128, 256, 512]
        elif backbone.lower() == 'resnet50':
            features = [64, 256, 512, 1024, 2048]
        elif backbone.lower() == 'resnet101':
            features = [64, 256, 512, 1024, 2048]
        elif backbone.lower() == 'resnet152':
            features = [64, 256, 512, 1024, 2048]
        else:
            print('Check your backbone again ^.^')
            return None
        
        self.decoder = nn.ModuleList([
            DecoderBLock(skip_channels=features[-2], x_channels=features[-1], kernel_size=3), 
            DecoderBLock(skip_channels=features[-3], x_channels=features[-2], kernel_size=3), 
            DecoderBLock(skip_channels=features[-4], x_channels=features[-3], kernel_size=3), 
            DecoderBLock(skip_channels=features[-5], x_channels=features[-4], kernel_size=3), 
        ]).to(device)
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=features[-5], out_channels=features[-5]//2, kernel_size=3, stride=1, padding="same"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=features[-5]//2, out_channels=1, kernel_size=1, stride=1, padding="same")
        ).to(device)

    def forward(self, x):
        b, c, h, w = x.shape
        enc = self.encoder(x) 
        block1, block2, block3, block4, block5 = enc[3], enc[5], enc[6], enc[7], enc[8]
        
        u1 = self.decoder[0](block4, block5)
        u2 = self.decoder[1](block3, u1)
        u3 = self.decoder[2](block2, u2)
        u4 = self.decoder[3](block1, u3)
        
        u4 = F.interpolate(u4, size=[h, w], mode='bilinear', align_corners=True)
        op = self.head(u4)

        return op

if __name__ == '__main__': 
    from prettytable import PrettyTable
    device = 'cpu'
    model = UNetResNet(device=device, backbone='resnet18')
    img = torch.randn(size=(5, 3, 192, 256)).to(device)
    pred = model(img).detach().numpy()[0][0]
    print(pred.shape)
    # print('--'*20)
    import matplotlib.pyplot as plt 
    
    plt.imshow(pred)
    plt.show()
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        # print(table)
        print(f"Total Trainable Params: {total_params:,}")
        return total_params
    
    count_parameters(model)