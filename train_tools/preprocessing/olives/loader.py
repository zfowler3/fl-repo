import pandas as pd
import numpy as np
from torchvision.transforms import transforms
from train_tools.preprocessing.olives.datasets import Olives_DiseaseDetection
import torch.utils.data as data

def get_patient_id(spreadsheet_root, mode='tr'):
    '''
    Returns unique patient IDs in given train/test split
    '''
    if mode == 'tr':
        sheet = pd.read_csv(spreadsheet_root + '/prime_trex_compressed.csv')
    else:
        sheet = pd.read_csv(spreadsheet_root + '/prime_trex_test_new.csv')
    ids = sheet['Eye_ID'].to_numpy()

    return np.unique(ids)

def get_patient_ids_by_visit(spreadsheet_root, max_val, mode='tr'):
    '''
    Returns unique patient IDs in given train/test split, where only patients have
    certain #s of visits are considered (max_val)
    '''
    if mode == 'tr':
        sheet = pd.read_csv(spreadsheet_root + '/prime_trex_compressed.csv')
    else:
        sheet = pd.read_csv(spreadsheet_root + '/prime_trex_test_new.csv')
    ids = sheet['Eye_ID'].to_numpy()
    unique_ids = np.unique(ids)
    new_ids = []
    labels = []
    for i in unique_ids:
        subsheet = sheet[sheet['Eye_ID'] == i].reset_index().iloc[:, 1:]
        max_visit = subsheet['Visit'].max()
        lab = subsheet['Label'].iloc[0]
        # if patient has at least 'max_val' number of visits, include it
        if max_visit > max_val:
            new_ids.append(i)
            labels.append(lab)

    return np.array(new_ids), np.array(labels)


def get_all_targets_olives(root, mode='tr', dataset_label=None):
    dataset = Olives_DiseaseDetection(root=root, mode=mode)
    all_targets = dataset.targets
    return all_targets

def _data_transforms():
    mean = .1706
    std = .2112
    data_transforms = {
        'train': transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    return data_transforms['train'], data_transforms['test']

def get_dataloader_olives(root, batch_size=50, dataidxs=None, dataset_label=None, mode='tr'):
    # Dataloaders for olives DR vs DME detection
    train_transform, valid_transform = _data_transforms()

    if mode == 'tr':
        dataset = Olives_DiseaseDetection(
            root, dataidxs=dataidxs, mode='tr', transforms=train_transform
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=5
        )

    else:
        dataset = Olives_DiseaseDetection(
            root, dataidxs=dataidxs, mode=mode, transforms=valid_transform
        )
        print('TEST dataset size: ', len(dataset))
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=5
        )

    return dataloader