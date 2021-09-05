import numpy as np

import torch
import torchvision

class MnistSampler:
    def __init__(self) -> None:
        self.mnist_dataset = torchvision.datasets.MNIST('./data/mnist', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ]))

    def sample(self, num):
        # samples an image of num from the MNIST dataset
        idxs = (self.mnist_dataset.targets == num).nonzero(as_tuple=True)[0]
        idx = np.random.choice(idxs)
        return self.mnist_dataset[idx][0] 


    def sample_digits(self, top, n_samples=1):
        images = torch.stack([self.sample(n) for _ in range(n_samples) for n in range(top)])
        labels = torch.LongTensor([n for _ in range(n_samples) for n in range(top)])
        return images, labels

    def sample_array(self, top):
        images = torch.stack([self.sample(n)  for n in range(top)])
        return images


if __name__ == '__main__':
    MnistSampler() # download the MNIST dataset