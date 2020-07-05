import torch.nn as nn
import torch
import numpy as np
import pickle as pkl

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')


def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1.
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    new_range = feature_range[1] - feature_range[0]
    x = feature_range[0] + x * new_range

    return x


def real_loss(D_out):
    # Calculates how close discriminator outputs are to being real.
    # create labels
    labels = torch.ones(D_out.size(0))

    # move to gpu if available
    if train_on_gpu:
        labels = labels.cuda()

    # binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()

    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    # Calculates how close discriminator outputs are to being fake.
    # create labels
    labels = torch.zeros(D_out.size(0))

    # move to gpu if available
    if train_on_gpu:
        labels = labels.cuda()

    # binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()

    loss = criterion(D_out.squeeze(), labels)
    return loss


def train(D, G, g_optimizer, d_optimizer, n_epochs, z_size, dataloader, model_name, print_every=50):

    # move models to GPU
    if train_on_gpu:
        D.cuda()
        G.cuda()

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size = 16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, _) in enumerate(dataloader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)

            # ===============================================
            #               TRAIN THE NETWORKS
            # ===============================================

            d_optimizer.zero_grad()

            # Train the discriminator on real and fake images
            if train_on_gpu:
                real_images = real_images.cuda()

            D_real = D.forward(real_images)
            d_real_loss = real_loss(D_real)

            # Generate fake images
            # create z input vector
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            # generate
            fake_images = G(z)

            # Compute loss on fake images
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake)

            # calculate total loss and perform backpropagation
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train the generator with an adversarial loss

            g_optimizer.zero_grad()

            # Generate fake images
            # create z input vector
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            # generate
            fake_images = G(z)

            # Compute loss on fake images
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake)

            # perform backpropagation
            g_loss.backward()
            g_optimizer.step()

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, n_epochs, d_loss.item(), g_loss.item()))

        ## AFTER EACH EPOCH##
        G.eval()  # for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()  # back to training mode

        if epoch % 5 == 0 and epoch != 0:
            torch.save(D, f"models/trained_D{model_name}_{epoch // 5}.pt")
            torch.save(G, f"models/trained_G{model_name}_{epoch // 5}.pt")

        torch.save(D, f"models/trained_D{model_name}.pt")
        torch.save(G, f"models/trained_G{model_name}.pt")

        # Save training generator samples
        with open('train_samples.pkl', 'wb') as f:
            pkl.dump(samples, f)

    # return losses
    return losses
