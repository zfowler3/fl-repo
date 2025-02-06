import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

from .datasets import CIFAR100_truncated


class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def _data_transforms_cifar100():
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),]
    )

    return train_transform, valid_transform


def get_all_targets_cifar100(root, mode='tr', dataset_label=None, class_complexity=None):
    if class_complexity is not None:
        dataset = CIFAR100_truncated(root=root, mode=mode, class_complexity=class_complexity)
        all_targets = dataset.targets
    else:
        dataset = CIFAR100_truncated(root=root, mode=mode)
        all_targets = dataset.targets

    return all_targets


def choose_batch(size, min_batch, max_batch):
    min_batch_size = max(min_batch, 2)
    bs = 2
    while bs <= min_batch_size or bs > max_batch:
        bs *= 2

    bs = min(bs, size)
    return bs

def get_dataloader_cifar100(root, batch_size=50, dataidxs=None, dataset_label=None, mode='tr', loaded_targets=None,
                            class_complexity=None):
    train_transform, valid_transform = _data_transforms_cifar100()

    # if mode == 'tr':
    #     while len(dataidxs) % batch_size == 1:
    #         batch_size += 1

    if mode == 'tr':
        dataset = CIFAR100_truncated(
            root, dataidxs, mode='tr', transform=train_transform, download=False, loaded_targets=loaded_targets,
            class_complexity=class_complexity
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=5
        )

    else:
        dataset = CIFAR100_truncated(
            root, dataidxs, mode=mode, transform=valid_transform, download=False, loaded_targets=loaded_targets,
            class_complexity=class_complexity
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=5
        )

    return dataloader
