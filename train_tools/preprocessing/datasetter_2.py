import numpy as np
import os
from .cifar10c.loader import get_all_targets_cifar10c, get_dataloader_cifar10c
from .fashion.loader import get_all_targets_fashion, get_dataloader_fashion
from .medmnist.loader import get_dataloader_medmnist, get_all_targets_medmnist
from .mnist.loader import get_all_targets_mnist, get_dataloader_mnist
from .cifar10.loader import get_all_targets_cifar10, get_dataloader_cifar10
from .cifar100.loader import get_all_targets_cifar100, get_dataloader_cifar100
from .cinic10.loader import get_all_targets_cinic10, get_dataloader_cinic10
from .tinyimagenet.loader import (
    get_all_targets_tinyimagenet,
    get_dataloader_tinyimagenet,
)

__all__ = ["data_distributer"]

DATA_INSTANCES = {
    "mnist": get_all_targets_mnist,
    "cifar10": get_all_targets_cifar10,
    "cifar100": get_all_targets_cifar100,
    "cinic10": get_all_targets_cinic10,
    "tinyimagenet": get_all_targets_tinyimagenet,
    "BloodMNIST": get_all_targets_medmnist,
    "OrganCMNIST": get_all_targets_medmnist,
    "OCTMNIST": get_all_targets_medmnist,
    "TissueMNIST": get_all_targets_medmnist,
    "OrganSMNIST": get_all_targets_medmnist,
    "CIFAR-10-C": get_all_targets_cifar10c,
    "fashion": get_all_targets_fashion
}

DATA_LOADERS = {
    "mnist": get_dataloader_mnist,
    "cifar10": get_dataloader_cifar10,
    "cifar100": get_dataloader_cifar100,
    "cinic10": get_dataloader_cinic10,
    "tinyimagenet": get_dataloader_tinyimagenet,
    "BloodMNIST": get_dataloader_medmnist,
    "DermaMNIST": get_dataloader_medmnist,
    "OrganCMNIST": get_dataloader_medmnist,
    "OrganSMNIST": get_dataloader_medmnist,
    "OCTMNIST": get_dataloader_medmnist,
    "TissueMNIST": get_dataloader_medmnist,
    "CIFAR-10-C": get_dataloader_cifar10c,
    "fashion": get_dataloader_fashion
}

## initial code only for cifar-10 dataset and only uses iid partitioning

def data_distributer(
    root,
    dataset_name,
    batch_size,
    n_clients,
    partition,
    save_folder,
    shift_type,
):


    if dataset_name == "cifar10":
        root = os.path.join(root, dataset_name)
    
    # in loader.py. for cifar10:
    # get_dataloader_cifar10(root, batch_size=50, dataidxs=None, dataset_label=None, mode='tr'):
    
    # gets all available classes for train samples
    all_targets = DATA_INSTANCES[dataset_name](root, dataset_label=dataset_name)
    # number of classes
    num_classes = len(np.unique(all_targets))
    print('Class count: ', num_classes)
    
    # ask Zoe :
    net_dataidx_map_test = None
    
    # initialize up local_loaders dictionary
    # at each round, there is a different noise type (& parameters)
    # no noise added at round 0
    
    # random noise is added
    # at 1 round, each client may have different kind of noise
    
    local_loaders = {
        i:{} for i in range(n_clients)
    }
    for j in range(n_clients):
        local_loaders[j] = {
            k: {"datasize":0, "train": None, "test": None, "test_size": 0, "noise": None, "parameters" = [] } for k in range(n_rounds)
        }
    # local_loaders = {client_index : {round_number: {data, ... noise} } }
    
    # creates the dictionary mapping client index to which specific images (by index) they get
    if partition.method == "iid":
        net_dataidx_map = iid_partition(all_targets, n_clients)
    else:
        raise NotImplementedError
    
    net_dataidx_map_test, net_dataidx_map = create_local(idxs=net_dataidx_map, all_targets=all_targets,
                                                             save_folder=save_folder)
    
    ## Distributing Local Client train and test data
    print(">>> Distributing client train data...")
    print(save_folder)
        
    # retrives the actual images
    for client_idx, dataidxs in net_dataidx_map.items():
        for k in range(n_rounds):
            local_loaders[client_idx][k]["datasize"] = len(dataidxs)
            local_loaders[client_idx][k]["train"] = DATA_LOADERS[dataset_name](
                root, mode='tr', batch_size=batch_size, dataidxs=dataidxs, dataset_label=dataset_name
                )
    
    # assigns noise
    for k in range(n_rounds):
        if k == 0:
            continue
        else:
            noise_type, parameters = get_noise_function() # randomize noise type and parameters
            local_loaders[client_idx][k]["noise"] = noise_type
            local_loaders[client_idx][k]["parameters"] = parameters
            local_loaders[client_idx][k]["train"] = apply_noise(noise_type, parameters, img_size=(32,32)) # apply noise to images in dataset
    
    
    ## test data            
    if net_dataidx_map_test is not None:
        print(">>> Distributing client test data...")
        for client_idx, dataidxs in net_dataidx_map_test.items():
            # Note: Must train mode if not wanting to use train set (here: local client test set is made from local client data)
            if dataset_name == 'CIFAR-10-C':
                all_test = []
                for j in range(len(dataidxs)):
                    cur_visit_idxs = dataidxs[j]
                    local_loaders[client_idx][j]["test_size"] = len(cur_visit_idxs)
                    local_loaders[client_idx][j]["test"] = DATA_LOADERS[dataset_name](
                        root, mode='tr', batch_size=batch_size, dataidxs=cur_visit_idxs, dataset_label=dataset_name,
                        shift_type=shift_type
                    )
                    all_test.append(local_loaders[client_idx][j]["test"])
                local_loaders[client_idx]["all_test"] = all_test
            else:
                local_testloader = DATA_LOADERS[dataset_name](
                    root, mode='tr', batch_size=batch_size, dataidxs=dataidxs, dataset_label=dataset_name
                )
                local_loaders[client_idx]["test"] = local_testloader
                local_loaders[client_idx]["test_size"] = len(dataidxs)
    
###

def get_noise_function():
    noise_types = ["gaussian","poisson","uniform","salt_and_pepper","periodic"]
    
    noise_type = noise_types[np.random(0,5)]
    if noise_type == "gaussian":
        mu = np.random(-1,1)
        var = np.random(0,5)
        parameters = [mu,var]

    elif noise_type == "poisson":
        lambda_param =  np.random(1,20)
        parameters = [lambda_param]
        
    elif noise_type == "salt_and_pepper":
        low_val = 0
    elif noise_type == "uniform":
        min_val = 
        max_val = 
        #noise = np.random.uniform(min_val, max_val, image.shape).astype(image.dtype)
    elif noise_type == "periodic":
        min
    return noise_type, parameters


## adding noise:
## skimage.util.random_noise(image, mode='gaussian', rng=None, clip=True, **kwargs)[source]
# modes: gaussian’, ‘poisson’ ‘s&p’, ‘speckle’
# is data a list of indexes
def apply_noise(client_idx, local_loaders, img_size):
    clean_data = local_loaders[client_idx][0]["train"]
    noisy_data = []
    if noise_type == "gaussian":
        for image in clean_data:
            

    elif noise_type == "poisson":
        lambda_param =  np.random(1,20)
        parameters = [lambda_param]
        
    elif noise_type == "salt_and_pepper":
        low_val = 0
    elif noise_type == "uniform":
        #noise = np.random.uniform(min_val, max_val, image.shape).astype(image.dtype)
    elif noise_type == "periodic":
