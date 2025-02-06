import torch.utils.data as data

import numpy as np
from torchvision.datasets import CIFAR10


class CIFAR10_C(data.Dataset):
    def __init__(self, root, dataidxs=None, mode='tr', transform=None, download=True, noise_type='gaussian_noise'):
        self.root = root
        self.dataidxs = dataidxs
        if mode == 'tr':
            train = True
        else:
            train = False
        self.train = train
        self.transform = transform
        self.download = download
        self.num_classes = 10
        self.mode = mode
        self.noise_type = noise_type
        self.data, self.targets = self._build_dataset()

    def _build_dataset(self):

        base_dataset = CIFAR10(
            self.root, self.train, self.transform, None, self.download
        )
        # Load noisy dataset
        if self.train:
            data = np.load(self.root + '/' + self.noise_type + '.npy')
        else:
            data = base_dataset.data

        # Get targets from original CIFAR-10 Dataset
        targets = np.array(base_dataset.targets)

        if self.train:
            # Cifar-10-C is corrupted cifar-10 test set
            base_dataset = CIFAR10(
                self.root, False, self.transform, None, self.download
            )
            targets = np.array(base_dataset.targets)
            targets = np.tile(targets, 5)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, targets, index

    def __len__(self):
        return len(self.data)