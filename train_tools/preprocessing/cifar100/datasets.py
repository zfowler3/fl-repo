import torch.utils.data as data
from torchvision.datasets import CIFAR100

import numpy as np

import numpy as np


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

class CIFAR100_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, mode='tr', transform=None, download=True, class_complexity=None,
                 loaded_targets=None):
        self.root = ('/').join((root.split('/'))[:-1])
        self.dataidxs = dataidxs
        if mode == 'tr':
            train = True
        else:
            train = False
        self.train = train
        self.transform = transform
        self.download = download

        self.data, self.targets = self._build_truncated_dataset()

        if class_complexity is None:
            self.num_classes = 100
        else:
            self.class_complexity = class_complexity
            self.original = 100
            self.num_classes = class_complexity
            if loaded_targets is not None:
                #print('LOADING PRIOR TARGETS DETERMINED')
                selected_classes = np.unique(loaded_targets)
                base_dataset = CIFAR100(
                    self.root, self.train, self.transform, None, self.download
                )

                data = base_dataset.data
                targets = np.array(base_dataset.targets)
                #print('cur: ', targets)
                inds = np.where(np.isin(targets, selected_classes))[0]

                self.targets = targets[inds]
                self.data = data[inds]

                if self.dataidxs is not None:
                    self.data = self.data[self.dataidxs]
                    self.targets = self.targets[self.dataidxs]
            else:
                self.data, self.targets = self._class_complexity()

        # self.targets = sparse2coarse(self.targets)
        # self.num_classes = 20

    def _build_truncated_dataset(self):
        base_dataset = CIFAR100(
            self.root, self.train, self.transform, None, self.download
        )

        data = base_dataset.data
        targets = np.array(base_dataset.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def _class_complexity(self):
        # select classes
        print(self.class_complexity)
        selected_classes = np.random.choice(np.arange(0, self.original), size=self.class_complexity, replace=False)
        inds = np.where(np.isin(self.targets, selected_classes))[0]

        targets = self.targets[inds]
        #print(targets)
        data = self.data[inds]

        return data, targets

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, targets, index

    def __len__(self):
        return len(self.data)
