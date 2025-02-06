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


def data_distributer(
    root,
    dataset_name,
    batch_size,
    n_clients,
    partition,
    save_folder,
    shift_type,
):
    """
    Distribute dataloaders for server and locals by the given partition method.
    """
    if dataset_name == 'cinic10':
        root = os.path.join(root, 'CINIC-10')
    elif dataset_name == 'tinyimagenet':
        root = os.path.join(root, 'tiny-imagenet-200')
    elif dataset_name != 'CIFAR-10-C':
        root = os.path.join(root, dataset_name.lower())
    else:
        root = os.path.join(root, dataset_name)

    # Get all available classes for train samples
    all_targets = DATA_INSTANCES[dataset_name](root, dataset_label=dataset_name)
    # Figure out number of classes
    num_classes = len(np.unique(all_targets))
    print('Class count: ', num_classes)

    net_dataidx_map_test = None

    if dataset_name != 'CIFAR-10-C':
        local_loaders = {
            i: {"datasize": 0, "train": None, "test": None, "test_size": 0} for i in range(n_clients)
        }
    else:
        local_loaders = {
            i: {} for i in range(n_clients)
        }
        # TODO: Cifar10c setup can be modified. Local loaders holds the data (train/test) for each client. In
        #  Federated Continual Learning, data across each client will shift over time. This will need to be accounted
        #  for here. Right now, I have set it up in a way such that each client has a separate train/test dictionary per
        #  noise level in the cifar10c dataset.
        noise_levels = 5
        for j in range(n_clients):
            local_loaders[j] = {
                i: {"datasize": 0, "train": None, "test": None, "test_size": 0} for i in range(noise_levels)
            }
            local_loaders[j]["all_test"] = None


    if dataset_name == 'CIFAR-10-C':
        net_dataidx_map = data_distribution_cifar10c(all_targets, n_clients, partition.alpha, num_classes, partition.method)
        # create local test set FROM net_dataidx_map
        net_dataidx_map_test, net_dataidx_map = local_data_distribution_shift(idxs=net_dataidx_map,
                                                                              all_targets=all_targets,
                                                                              save_folder=save_folder)
    else:
        if partition.method == "centralized":
            net_dataidx_map = centralized_partition(all_targets)
        elif partition.method == "iid":
            net_dataidx_map = iid_partition(all_targets, n_clients)
        elif partition.method == 'dirichlet':
            net_dataidx_map = partition_class_samples_with_dirichlet_distribution(alpha=partition.alpha, client_num=n_clients,
                                                                                  targets=all_targets, class_num=num_classes)
        else:
            raise NotImplementedError

        net_dataidx_map_test, net_dataidx_map = create_local(idxs=net_dataidx_map, all_targets=all_targets,
                                                             save_folder=save_folder)

    # Distributing Local Client train and test data
    print(">>> Distributing client train data...")
    print(save_folder)

    for client_idx, dataidxs in net_dataidx_map.items():
        if dataset_name == 'CIFAR-10-C':
            for j in range(len(dataidxs)):
                cur_idxs = dataidxs[j]
                local_loaders[client_idx][j]["datasize"] = len(cur_idxs)
                local_loaders[client_idx][j]["train"] = DATA_LOADERS[dataset_name](
                    root, mode='tr', batch_size=batch_size, dataidxs=cur_idxs, dataset_label=dataset_name,
                    shift_type=shift_type
                )
        else:
            local_loaders[client_idx]["datasize"] = len(dataidxs)
            local_loaders[client_idx]["train"] = DATA_LOADERS[dataset_name](
                root, mode='tr', batch_size=batch_size, dataidxs=dataidxs, dataset_label=dataset_name
            )

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

    ################################################################################################################
    # Global Dataloader (For testing generalization)
    ###############################################################################################################
    test_global_loader = DATA_LOADERS[dataset_name](root, mode='te', batch_size=batch_size, dataset_label=dataset_name)
    global_loaders = {
        "test": test_global_loader,
        "test_size": int(len(test_global_loader)*batch_size)
    }

    data_distributed = {
        "global": global_loaders,
        "local": local_loaders,
        "num_classes": num_classes,
    }

    return data_distributed


def centralized_partition(all_targets):
    labels = all_targets
    tot_idx = np.arange(len(labels))
    net_dataidx_map = {}

    tot_idx = np.array(tot_idx)
    np.random.shuffle(tot_idx)
    net_dataidx_map[0] = tot_idx

    return net_dataidx_map


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


def data_distribution_cifar10c(all_targets, n_clients, alpha, num_classes, method, in_order=True):
    noise_levels = 5
    num = 10000
    net_dataidx_map = {}
    # For first (clean) imgs, do dirichlet
    modified_targets = all_targets[:num]
    print('modified targets length: ', len(modified_targets))
    if method == 'dirichlet':
        selected_idxs = partition_class_samples_with_dirichlet_distribution(alpha=alpha, client_num=n_clients,
                                                                            targets=modified_targets, class_num=num_classes)
    else:
        selected_idxs = iid_partition(modified_targets, n_clients)

    if in_order:
        for j in range(n_clients):
            noise = {}
            count = 0
            print('--------------------------')
            for level in range(noise_levels):
                idxs = selected_idxs[j] + count
                noise[level] = idxs
                count += num
            net_dataidx_map[j] = noise
    else:
        print('not in order!')
        # clients noise levels are out of order randomly (i.e., client 1 may start with level 1 --> level 5 --> etc, client 2 might be level 0 --> level 1 ...)
        for j in range(n_clients):
            noise = {}
            nums = np.array([0, 10000, 20000, 30000, 40000])
            for level in range(noise_levels):
                print(nums)
                idx = np.random.choice(np.arange(len(nums)))
                count = nums[idx]
                nums = np.delete(nums, idx)
                noise[level] = selected_idxs[j] + count
            net_dataidx_map[j] = noise
    return net_dataidx_map

def local_data_distribution_shift(idxs, all_targets, amount=0.20, save_folder='/home/zoe/Dropbox (GhassanGT)/Zoe/InSync/PhDResearch/Code/Results/NeurIPS2024/'):
    net_dataidx_test = {}
    for i in range(len(idxs)):
        current_client_idxs = idxs[i]
        test = {}
        tr = {}
        # SAME TEST PARTITION USED ACROSS DIFFERENT NOISE LEVELS
        cur_idxs = current_client_idxs[0]
        local_test_amount = int(len(current_client_idxs[0]) * amount)
        # get unique classes
        classes = all_targets[cur_idxs]
        unique_classes = np.unique(classes)
        num_classes = len(unique_classes)
        per_class = int(local_test_amount / num_classes)
        # for client's total num of classes, select local test idxs
        test_idxs = np.array([])
        for c in range(len(unique_classes)):
            curr_class = unique_classes[c]
            class_idxs = cur_idxs[np.where(all_targets[cur_idxs] == curr_class)]
            try:
                test_idxs_class = np.random.choice(class_idxs, size=per_class, replace=False)
            except ValueError:
                test_idxs_class = np.random.choice(class_idxs, size=int(amount * len(class_idxs)), replace=False)
            test_idxs = np.concatenate((test_idxs, test_idxs_class))

        # if test idxs are still empty, randomly select samples
        if len(test_idxs) == 0:
            test_idxs = np.random.choice(cur_idxs, size=int(amount * len(cur_idxs)),
                                         replace=False)
        # For each client, get idxs corresponding to local test set
        for j in range(len(current_client_idxs)):
            actual_idxs = np.where(np.isin(cur_idxs, test_idxs))[0]
            test[j] = current_client_idxs[j][actual_idxs].astype(int)
            new_train = np.where(np.isin(cur_idxs, test_idxs, invert=True))[0]
            tr[j] = current_client_idxs[j][new_train].astype(int)

        idxs[i] = tr
        net_dataidx_test[i] = test

    np.save(save_folder + 'test_idxs.npy', net_dataidx_test, allow_pickle=True)
    np.save(save_folder + 'train_idxs.npy', idxs, allow_pickle=True)
    return net_dataidx_test, idxs

def create_local(idxs, all_targets, amount=0.20, save_folder='/home/zoe/Dropbox (GhassanGT)/Zoe/InSync/PhDResearch/Code/Results/NeurIPS2024/'):
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

    np.save(save_folder + 'test_idxs.npy', net_dataidx_test, allow_pickle=True)
    np.save(save_folder + 'train_idxs.npy', idxs, allow_pickle=True)
    return net_dataidx_test, idxs


def dirichlet(N, alpha, client_num, idx_batch, idx_k):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array(
        [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
    )
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [
        idx_j + idx.tolist()
        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    ]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size

def partition_class_samples_with_dirichlet_distribution(
    alpha, client_num, targets, class_num
):
    net_dataidx_map = {}
    min_size = 0
    min_require_size = 10
    N = len(targets)
    # print(N)
    # print(class_num)
    # print(targets)

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(client_num)]
        for k in np.unique(targets):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, client_num))

            # get the index in idx_k according to the dirichlet distribution
            proportions = np.array(
                [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # generate the batch list for each client
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(len(idx_batch)):
        idx_batch[j] = np.array(idx_batch[j]).astype('int')

    for j in range(client_num):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map

