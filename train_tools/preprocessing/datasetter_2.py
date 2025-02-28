import numpy as np
import os
import matplotlib as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

__all__ = ["data_distributer"]
'''
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
'''
## initial code only for cifar-10 dataset and only uses iid partitioning

def data_distributer(
    root,
    dataset_name,
    batch_size,
    n_clients,
    n_rounds,
    partition,
    save_folder,
    shift_type,
):


    if dataset_name == "cifar10":
        root = os.path.join(root, dataset_name)
    
    # get train set
    train_set = torchvision.datasets.CIFAR10(root="/.data", train=True, download=True, transform=transforms.ToTensor())
    train_images = train_set.data
    train_labels = np.array(train_set.targets)
    
    # get test set
    test_set = torchvision.datasets.CIFAR10(root="/.data", train=False, download=True, transform=transforms.ToTensor())
    test_images = test_set.data #.numpy()
    test_labels = np.array(test_set.targets)
    
    # entire dataset
    all_labels = np.append(train_labels,test_labels) # 60000 x 0
    all_images = np.append(train_images,test_images,axis=0) # 60000 x 32x32x3
    
    
    # gets all available classes for train samples
    num_classes = len(np.unique(all_labels))  # number of classes
    print('Class count: ', num_classes)
    
    
    ### initialize train and test data by indices
    net_dataidx_map_test = None
    
    # initialize local loaders dictionary
    local_loaders = {
    i:{} for i in range(n_clients)
    }
    
    for j in range(n_clients):
        local_loaders[j] = {
            k: {"n_rounds": n_rounds, "datasize":0, "train": {"images":[],"labels":[]}, "test": {"images":[],"labels":[]}, "test_size": 0, "noise": None, "parameters": [] } for k in range(n_rounds)
        }
    
    
    net_dataidx_map = iid_partition(all_labels, n_clients)
    net_dataidx_map_test, net_dataidx_map = create_local(idxs=net_dataidx_map, all_targets=all_labels, save_folder=save_folder)
    
    
    # load train data for each client as numpy array
    for client, index_list in net_dataidx_map.items():
        num_labels = len(index_list)
        client_imgs = np.zeros((num_labels,32,32,3))
        client_labels = np.zeros((num_labels,))
        for i, index in enumerate(index_list):
            client_imgs[i] = all_images[index]
            client_labels[i] = all_labels[index]
        local_loaders[client][0]["train"]["images"] = client_imgs.astype(np.uint8)
        local_loaders[client][0]["train"]["labels"] = client_labels
    
    # load test data
    for client, index_list in net_dataidx_map_test.items():
        num_labels = len(index_list)
        client_imgs = np.zeros((num_labels,32,32,3))
        client_labels = np.zeros((num_labels,))
  
        for i, index in enumerate(index_list):
            client_imgs[i] = all_images[index]
            client_labels[i] = all_labels[index]
        local_loaders[client][0]["test"]["images"] = client_imgs.astype(np.uint8)
        local_loaders[client][0]["test"]["labels"] = client_labels
        
    
    ## Apply noise to train and test sets
    for client_idx in range(n_clients):
        client_labels = local_loaders[client_idx][0]["train"]["labels"]
        test_labels = local_loaders[client_idx][0]["test"]["labels"]
        num_labels = len(client_labels) # load labels (doesnt change per round)
        num_test = len(test_labels)
  
        # for each round after 1st round, add random noise
        for k in range(n_rounds):
            if k == 0:
                continue
            
            prev_dataset = local_loaders[client_idx][k-1]["train"]["images"] # loads images
            noisy_dataset = np.zeros((num_labels,32,32,3)) # initialize empty dataset
    
            noise_type, parameters = get_noise_function() # get random noise
            local_loaders[client_idx][k]["noise"] = noise_type
            local_loaders[client_idx][k]["parameters"] = parameters # store in local loaders
    
            for i, img in enumerate(prev_dataset):
                noisy_dataset[i] = apply_noise(img,noise_type,parameters) # add noise to previous round's dataset
            local_loaders[client_idx][k]["train"]["images"] = noisy_dataset
            local_loaders[client_idx][k]["train"]["labels"] = client_labels

            # apply noise to test data
            prev_test = local_loaders[client_idx][k-1]["test"]["images"] # load test images
            noisy_test = np.zeros((num_test,32,32,3)) # initialize empty test set
    
            for i, test_img in enumerate(prev_test):
                noisy_test[i] = apply_noise(test_img,noise_type,parameters) # add noise to previous round's dataset
            local_loaders[client_idx][k]["test"]["images"] = noisy_test
            local_loaders[client_idx][k]["test"]["labels"] = test_labels

def get_noise_function():
  return "gaussian", [0,1]

def apply_noise(clean_img, noise_type, parameters):
  if noise_type == "gaussian":
    mu = parameters[0]
    sigma = parameters[1]
  noisy_img = clean_img + np.random.normal(0,1,(32,32,3)).astype(np.float32)
  noisy_img = np.nan_to_num(noisy_img)
  noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
  return noisy_img

def iid_partition(all_targets, n_clients):
    labels = all_targets
    length = int(len(labels) / n_clients)
    tot_idx = np.arange(len(labels))
    net_dataidx_map = {}

    for client_idx in range(n_clients):
        np.random.shuffle(tot_idx)
        data_idxs = tot_idx[:length]
        tot_idx = tot_idx[length:]
        net_dataidx_map[client_idx] = np.array(data_idxs)

    return net_dataidx_map

    
def create_local(idxs, all_targets, amount=0.20, save_folder):
    # Creating local test clients from partitioned data
    n_clients = len(idxs)
    net_dataidx_test = {i: np.array([], dtype="int64") for i in range(n_clients)}
    for i in range(len(idxs)):
        current_client_idxs = idxs[i]
        local_test_amount = int(len(current_client_idxs)*amount)
        # get unique classes
        classes = all_targets[current_client_idxs]
        unique_classes = np.unique(classes)
        num_classes = len(unique_classes)
        per_class = int(local_test_amount/num_classes)
        # for client's total num of classes, select local test idxs
        test_idxs = np.array([])
        for c in range(len(unique_classes)):
            curr_class = unique_classes[c]
            class_idxs = current_client_idxs[np.where(all_targets[current_client_idxs] == curr_class)]
            try:
                test_idxs_class = np.random.choice(class_idxs, size=per_class, replace=False)
            except ValueError:
                test_idxs_class = np.random.choice(class_idxs, size=int(amount*len(class_idxs)), replace=False)
            test_idxs = np.concatenate((test_idxs, test_idxs_class))
        # if test idxs are still empty, randomly select samples
        if len(test_idxs) == 0:
            test_idxs = np.random.choice(current_client_idxs, size=int(amount*len(current_client_idxs)), replace=False)
        # For each client, get idxs corresponding to local test set
        net_dataidx_test[i] = test_idxs.astype(int)
        new_train = np.where(np.isin(current_client_idxs, test_idxs, invert=True))[0]
        idxs[i] = current_client_idxs[new_train].astype(int)

    #np.save(save_folder + 'test_idxs.npy', net_dataidx_test, allow_pickle=True)
    #np.save(save_folder + 'train_idxs.npy', idxs, allow_pickle=True)
    return net_dataidx_test, idxs
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
