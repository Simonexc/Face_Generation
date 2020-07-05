import torch.nn as nn
from .model import Discriminator, Generator


def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution
    with mean = 0, std dev = 0.02.
    """

    classname = m.__class__.__name__

    if classname == "Conv2d" or classname == "ConvTranspose2d" or classname == "Linear":
        nn.init.normal_(m.weight.data, 0, 0.02)


def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    print(D)
    print()
    print(G)

    return D, G
