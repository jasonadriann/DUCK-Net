import torch
import torch.nn as nn
import torch.nn.functional as F

from CustomLayers.PyTorchConvBlock2D import ConvBlock2D

class DuckNet(nn.Module):
    def __init__(self, img_height, img_width, input_channels, out_classes, starting_filters):
        super(DuckNet, self).__init__()

        self.starting_filters = starting_filters

        self.conv1 = nn.Conv2d(input_channels, starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding=0)

        self.t0 = ConvBlock2D(input_channels, starting_filters, 'duckv2', repeat=1)

        self.l1i = nn.Conv2d(starting_filters, starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.t1 = ConvBlock2D(starting_filters * 2, starting_filters * 2, 'duckv2', repeat=1)

        self.l2i = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.t2 = ConvBlock2D(starting_filters * 4, starting_filters * 4, 'duckv2', repeat=1)

        self.l3i = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.t3 = ConvBlock2D(starting_filters * 8, starting_filters * 8, 'duckv2', repeat=1)

        self.l4i = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding=0)
        self.t4 = ConvBlock2D(starting_filters * 16, starting_filters * 16, 'duckv2', repeat=1)

        self.l5i = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding=0)
        self.t51 = ConvBlock2D(starting_filters * 32, starting_filters * 32, 'resnet', repeat=2)
        self.t53 = ConvBlock2D(starting_filters * 32, starting_filters * 16, 'resnet', repeat=2)

        self.l5o = nn.Upsample(scale_factor=2, mode='nearest')
        self.c4 = ConvBlock2D(starting_filters * 16, starting_filters * 8, 'duckv2', repeat=1)

        self.l4o = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3 = ConvBlock2D(starting_filters * 8, starting_filters * 4, 'duckv2', repeat=1)

        self.l3o = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2 = ConvBlock2D(starting_filters * 4, starting_filters * 2, 'duckv2', repeat=1)

        self.l2o = nn.Upsample(scale_factor=2, mode='nearest')
        self.c1 = ConvBlock2D(starting_filters * 2, starting_filters, 'duckv2', repeat=1)

        self.l1o = nn.Upsample(scale_factor=2, mode='nearest')
        self.c0 = ConvBlock2D(starting_filters, starting_filters, 'duckv2', repeat=1)

        self.output = nn.Conv2d(starting_filters, out_classes, kernel_size=1)

    def forward(self, x):
        p1 = self.conv1(x)
        p2 = self.conv2(p1)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)
        p5 = self.conv5(p4)

        t0 = self.t0(x)

        l1i = self.l1i(t0)
        s1 = l1i + p1
        t1 = self.t1(s1)

        l2i = self.l2i(t1)
        s2 = l2i + p2
        t2 = self.t2(s2)

        l3i = self.l3i(t2)
        s3 = l3i + p3
        t3 = self.t3(s3)

        l4i = self.l4i(t3)
        s4 = l4i + p4
        t4 = self.t4(s4)

        l5i = self.l5i(t4)
        s5 = l5i + p5
        t51 = self.t51(s5)
        t53 = self.t53(t51)

        l5o = self.l5o(t53)
        c4 = l5o + t4
        q4 = self.c4(c4)

        l4o = self.l4o(q4)
        c3 = l4o + t3
        q3 = self.c3(c3)

        l3o = self.l3o(q3)
        c2 = l3o + t2
        q6 = self.c2(c2)

        l2o = self.l2o(q6)
        c1 = l2o + t1
        q1 = self.c1(c1)

        l1o = self.l1o(q1)
        c0 = l1o + t0
        z1 = self.c0(c0)

        output = self.output(z1)

        return output