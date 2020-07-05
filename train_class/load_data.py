from torchvision import datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch


def get_dataloader(batch_size, image_size, data_dir='train_face/'):

    # resize and convert to Tensor
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor()])

    dataset = datasets.ImageFolder(data_dir, transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def show_images(dataloader):
    dataiter = iter(dataloader)
    images, _ = dataiter.next()  # _ for no labels

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(20, 4))
    plot_size = 20
    for idx in np.arange(plot_size):
        ax = fig.add_subplot(2, plot_size / 2, idx + 1, xticks=[], yticks=[])
        imshow(images[idx])
