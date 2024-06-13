import torch
import torch.nn as nn
import torch.nn.functional as F

# Define kernel_initializer equivalent in PyTorch
def he_uniform(tensor):
    nn.init.kaiming_uniform_(tensor, nonlinearity='relu')

class ConvBlock2D(nn.Module):
    def __init__(self, input_channels, filters, block_type, repeat=1, dilation_rate=1, size=3, padding='same'):
        super(ConvBlock2D, self).__init__()
        self.input_channels = input_channels
        self.filters = filters
        self.block_type = block_type
        self.repeat = repeat
        self.dilation_rate = dilation_rate
        self.size = size
        self.padding = padding
        self.kernel_initializer = he_uniform
        
        self.blocks = nn.ModuleList()
        
        for i in range(repeat):

            input_channels = input_channels if i == 0 else None

            if block_type == 'separated':
                self.blocks.append(SeparatedConv2DBlock(input_channels, filters, size, padding))
            elif block_type == 'duckv2':
                self.blocks.append(Duckv2Conv2DBlock(input_channels, filters, size))
            elif block_type == 'midscope':
                self.blocks.append(MidscopeConv2DBlock(input_channels, filters))
            elif block_type == 'widescope':
                self.blocks.append(WidescopeConv2DBlock(input_channels, filters))
            elif block_type == 'resnet':
                self.blocks.append(ResnetConv2DBlock(input_channels, filters, dilation_rate))
            elif block_type == 'conv':
                self.blocks.append(nn.Conv2d(input_channels, filters, filters, kernel_size=size, padding=padding, dilation=dilation_rate))
                self.blocks.append(nn.ReLU())
            elif block_type == 'double_convolution':
                self.blocks.append(DoubleConvolutionWithBatchNormalization(input_channels, filters, dilation_rate))
            else:
                raise ValueError(f"Unknown block type: {block_type}")

    def forward(self, x):
        result = x
        for block in self.blocks:
            result = block(result)
        return result
    
class Duckv2Conv2DBlock(nn.Module):
    def __init__(self, input_channels, filters, size):
        super(Duckv2Conv2DBlock, self).__init__()
        if input_channels is None:
            input_channels = filters

        self.bn1 = nn.BatchNorm2d(input_channels)
        self.widescope = WidescopeConv2DBlock(input_channels, filters)
        self.midscope = MidscopeConv2DBlock(input_channels, filters)
        self.resnet1 = ResnetConv2DBlock(input_channels, filters)
        self.resnet2 = ConvBlock2D(input_channels, filters, 'resnet', repeat=2)
        self.resnet3 = ConvBlock2D(input_channels, filters, 'resnet', repeat=3)
        self.separated = SeparatedConv2DBlock(input_channels, filters, size, padding='same')
        self.bn2 = nn.BatchNorm2d(filters)
        
    def forward(self, x):
        x = self.bn1(x)
        x1 = self.widescope(x)
        x2 = self.midscope(x)
        x3 = self.resnet1(x)
        x4 = self.resnet2(x)
        x5 = self.resnet3(x)
        x6 = self.separated(x)
        x = x1 + x2 + x3 + x4 + x5 + x6
        x = self.bn2(x)
        return x

class SeparatedConv2DBlock(nn.Module):
    def __init__(self, input_channels, filters, size=3, padding='same'):
        super(SeparatedConv2DBlock, self).__init__()
        if input_channels is None:
            input_channels = filters

        self.conv1 = nn.Conv2d(input_channels, filters, kernel_size=(1, size), padding=padding)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(size, 1), padding=padding)
        self.bn2 = nn.BatchNorm2d(filters)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        return x

class MidscopeConv2DBlock(nn.Module):
    def __init__(self, input_channels, filters):
        super(MidscopeConv2DBlock, self).__init__()
        if input_channels is None:
            input_channels = filters
    
        self.conv1 = nn.Conv2d(input_channels, filters, kernel_size=3, padding='same', dilation=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding='same', dilation=2)
        self.bn2 = nn.BatchNorm2d(filters)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        return x

class WidescopeConv2DBlock(nn.Module):
    def __init__(self, input_channels, filters):
        super(WidescopeConv2DBlock, self).__init__()
        if input_channels is None:
            input_channels = filters
        
        self.conv1 = nn.Conv2d(input_channels, filters, kernel_size=3, padding='same', dilation=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding='same', dilation=2)
        self.bn2 = nn.BatchNorm2d(filters)
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=3, padding='same', dilation=3)
        self.bn3 = nn.BatchNorm2d(filters)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        return x

class ResnetConv2DBlock(nn.Module):
    def __init__(self, input_channels, filters, dilation_rate=1):
        super(ResnetConv2DBlock, self).__init__()
        if input_channels is None:
            input_channels = filters

        self.conv1 = nn.Conv2d(input_channels, filters, kernel_size=1, padding='same', dilation=dilation_rate)
        self.conv2 = nn.Conv2d(input_channels, filters, kernel_size=3, padding='same', dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(filters)
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=3, padding='same', dilation=dilation_rate)
        self.bn3 = nn.BatchNorm2d(filters)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x_final = x + x1
        return x_final

class DoubleConvolutionWithBatchNormalization(nn.Module):
    def __init__(self, input_channels, filters, dilation_rate=1):
        super(DoubleConvolutionWithBatchNormalization, self).__init__()
        if input_channels is None:
            input_channels = filters

        self.conv1 = nn.Conv2d(input_channels, filters, kernel_size=3, padding='same', dilation=dilation_rate)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding='same', dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(filters)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        return x
