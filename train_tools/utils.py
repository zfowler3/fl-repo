from .models import *
from .models.resnet_old import *

__all__ = ["create_models"]

# This file creates models

MODELS = {
    "fedavg_mnist": fedavgnet.FedAvgNetMNIST,
    "fedavg_cifar": fedavgnet.FedAvgNetCIFAR,
    "fashion": fedavgnet.FedAvgNetFashion,
    "fedavg_tiny": fedavgnet.FedAvgNetTiny,
    "res10": resnet10,
    "res18": resnet18,
    "res34": resnet34,
    "medmnist": fedavgnet.MedMNISTNet,
    "olives": fedavgnet.olivesNet
}

NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "CIFAR-10-C": 10,
    "cifar100": 100,
    "cinic10": 10,
    "tinyimagenet": 200,
    "BloodMNIST": 8,
    "DermaMNIST": 7,
    "OrganCMNIST": 11,
    "OrganSMNIST": 11,
    "OCTMNIST": 4,
    "TissueMNIST": 8,
    "fashion": 10,
    "olives": 2
}


def create_models(model_name, dataset_name):
    """Create a network model"""

    num_classes = NUM_CLASSES[dataset_name]

    # Datasets that are black and white
    if dataset_name == 'OrganCMNIST':
        ch = 1
    elif dataset_name == 'OrganSMNIST':
        ch = 1
    elif dataset_name == 'TissueMNIST':
        ch = 1
    elif dataset_name == 'OCTMNIST':
        ch = 1
    elif dataset_name == 'olives':
        ch = 1
    elif dataset_name == 'mnist':
        ch = 1
    elif dataset_name == 'fashion':
        ch = 1
    elif dataset_name == 'seismic':
        ch = 1
    else:
        # RGB
        ch = 3

    model = MODELS[model_name](num_classes=num_classes, in_channels=ch)

    return model
