from torchvision.transforms import transforms
import torch.utils.data as data
from train_tools.preprocessing.fashion.datasets import FashionMNISTData


def _data_transforms_fashion():
    MNIST_MEAN = [0.5]
    MNIST_STD = [0.5]

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            # transforms.RandomCrop(28, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )

    valid_transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD),]
    )

    return train_transform, valid_transform


def get_all_targets_fashion(root, dataset_label=None, mode='tr'):
    dataset = FashionMNISTData(root=root, mode=mode)
    all_targets = dataset.targets
    return all_targets


def get_dataloader_fashion(root, batch_size=50, dataidxs=None, dataset_label=None, mode='tr'):
    train_transform, valid_transform = _data_transforms_fashion()

    if mode == 'tr':
        dataset = FashionMNISTData(
            root, dataidxs, mode=mode, transform=train_transform, download=True
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=5
        )

    else:
        dataset = FashionMNISTData(
            root, dataidxs, mode=mode, transform=valid_transform, download=True
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=5
        )

    return dataloader