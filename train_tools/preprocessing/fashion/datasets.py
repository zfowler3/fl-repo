from torchvision.datasets import FashionMNIST
import torch.utils.data as data
import numpy as np

class FashionMNISTData(data.Dataset):
    def __init__(self, root, dataidxs=None, mode='tr', transform=None, download=True):
        self.root = ('/').join((root.split('/'))[:-1]) + '/'
        self.dataidxs = dataidxs
        if mode == 'tr':
            train = True
        else:
            train = False

        self.train = train
        self.transform = transform
        self.download = download
        self.num_classes = 10

        self.data, self.targets = self._build_truncated_dataset()

    def _build_truncated_dataset(self):
        base_dataset = FashionMNIST(
            self.root, self.train, self.transform, None, self.download
        )

        data = base_dataset.data
        targets = np.array(base_dataset.targets)

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