import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import train_class.load_data as loading
from train_class.model import Discriminator, Generator
from train_class.load_model import build_network
from train_class.train_model import train

img_size = 128
data_dir = 'train_face/'
model_name = f'{img_size}_3'

batch_size = 128

d_conv_dim = 64
g_conv_dim = 100
z_size = 100

lr = 0.0004
betas = [0.5, 0.999]

n_epochs = 5
on_load = True

if __name__ == "__main__":
    # load data
    celeba_train_loader = loading.get_dataloader(batch_size, img_size, data_dir)
    # create networks
    D, G = build_network(d_conv_dim, g_conv_dim, z_size)

    if on_load:
        # load the existing models
        D = torch.load(f"models/trained_D{model_name}.pt")
        G = torch.load(f"models/trained_G{model_name}.pt")

    # set optimizers
    d_optimizer = optim.Adam(D.parameters(), lr, betas)
    g_optimizer = optim.Adam(G.parameters(), lr, betas)

    # train networks
    losses = train(D, G, g_optimizer, d_optimizer, n_epochs, z_size, celeba_train_loader, model_name)

    # display losses
    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
    plt.plot(losses.T[1], label='Generator', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
