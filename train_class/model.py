import torch.nn as nn


def convolutional_layer(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, apply_lrelu=True):
    layers = []

    # normal convolutional layer
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(conv_layer)

    if batch_norm:
        # add batch normalization layer
        layers.append(nn.BatchNorm2d(out_channels))

    if apply_lrelu:
        # add leaky ReLU activation function
        layers.append(nn.LeakyReLU(0.2))

    return nn.Sequential(*layers)


def transposed_convolutional(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True,
                             apply_relu=True):
    layers = []

    # normal convolutional layer
    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(transpose_conv_layer)

    if batch_norm:
        # add batch normalization layer
        layers.append(nn.BatchNorm2d(out_channels))

    if apply_relu:
        # add leaky ReLU activation function
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        # Initialize the Discriminator Module

        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim

        # complete init function
        self.conv1 = convolutional_layer(3, conv_dim, 4, batch_norm=False)  # 128x128 -> 64x64
        self.conv2 = convolutional_layer(conv_dim, conv_dim * 2, 4)  # 64x64 -> 32x32
        self.conv3 = convolutional_layer(conv_dim * 2, conv_dim * 4, 4)  # 32x32 -> 16x16
        self.conv4 = convolutional_layer(conv_dim * 4, conv_dim * 8, 4)  # 16x16 -> 8x8
        self.conv5 = convolutional_layer(conv_dim * 8, conv_dim * 16, 4)  # 8x8 -> 4x4

        self.fc = nn.Linear(conv_dim * 16 * 4 * 4, 1)

    def forward(self, x):
        # define feedforward behavior

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # flatten
        x = x.view(-1, self.conv_dim * 16 * 4 * 4)
        x = self.fc(x)

        return x


class Generator(nn.Module):

    def __init__(self, z_size, conv_dim):
        # Initialize the Generator Module

        super(Generator, self).__init__()

        self.conv_dim = conv_dim
        # complete init function
        self.fc = nn.Linear(z_size, 2 * 2 * conv_dim * 32)

        self.trans_conv1 = transposed_convolutional(conv_dim * 32, conv_dim * 16, 4)  # 2x2 -> 4x4
        self.trans_conv2 = transposed_convolutional(conv_dim * 16, conv_dim * 8, 4)  # 4x4 -> 8x8
        self.trans_conv3 = transposed_convolutional(conv_dim * 8, conv_dim * 4, 4)  # 8x8 -> 16x16
        self.trans_conv4 = transposed_convolutional(conv_dim * 4, conv_dim * 2, 4)  # 16x16 -> 32x32
        self.trans_conv5 = transposed_convolutional(conv_dim * 2, conv_dim, 4)  # 32x32 -> 64x64
        self.trans_conv6 = transposed_convolutional(conv_dim, 3, 4, batch_norm=False,
                                                    apply_relu=False)  # 64x64 -> 128x128

        self.tanh = nn.Tanh()

    def forward(self, x):
        # Forward propagation of the neural network

        # define feedforward behavior
        x = self.fc(x)

        # reshape to 2x2 image
        x = x.view(-1, self.conv_dim * 32, 2, 2)  # (batch_size, depth, 2, 2)

        x = self.trans_conv1(x)
        x = self.trans_conv2(x)
        x = self.trans_conv3(x)
        x = self.trans_conv4(x)
        x = self.trans_conv5(x)
        x = self.trans_conv6(x)
        x = self.tanh(x)

        return x
