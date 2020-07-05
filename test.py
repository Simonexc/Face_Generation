from train import d_conv_dim, g_conv_dim, z_size, model_name, img_size
import torch
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from train_class.load_model import Discriminator, Generator
from train_class.load_model import build_network
import torch.nn.functional as F


# load pretrained models (D -> discriminator, G -> generator)
D, G = build_network(d_conv_dim, g_conv_dim, z_size)
D = torch.load(f"models/trained_D{model_name}.pt")
G = torch.load(f"models/trained_G{model_name}.pt")


# show sample generated images from specific epoch
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / 2).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((img_size, img_size, 3)))


# generates new faces
def generate():
    sample_size = 16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    fixed_z = fixed_z.cuda()
    G.eval()  # for generating samples
    samples_z = G(fixed_z)
    G.train()  # back to training mode

    view_samples(0, [samples_z])


# generates new faces and picks the one that is assessed as the most real by the discriminator
def intelligent_generate():
    sample_size = 16

    G.eval()  # for generating samples
    max_val = 0
    while max_val < 0.3:
        fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
        fixed_z = torch.from_numpy(fixed_z).float()
        fixed_z = fixed_z.cuda()
        samples_z = G(fixed_z)
        predictions = F.sigmoid(D(samples_z)).cpu()
        predictions = predictions.detach().numpy()
        predictions = predictions.flatten()
        max_val = np.max(predictions)
        id = np.where(predictions == max_val)[0][0]

    G.train() # back to training mode
    fig, axes = plt.subplots(figsize=(1, 1), nrows=1, ncols=1, sharey=True, sharex=True)

    img = samples_z[id].cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = ((img + 1) * 255 / 2).astype(np.uint8)
    axes.imshow(img.reshape((img_size, img_size, 3)))

    view_samples(0, [samples_z])


# opens samples saved during training for each epoch
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)

#view_samples(0, samples)
#generate()
intelligent_generate()
