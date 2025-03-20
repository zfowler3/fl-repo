import pandas as pd
import numpy as np

def get_patient_id(spreadsheet_root, mode='tr'):
    '''
    Returns unique patient IDs in given train/test split
    '''
    spreadsheet_root = spreadsheet_root + '/spreadsheets/'
    if mode == 'tr':
        sheet = pd.read_csv(spreadsheet_root + 'prime_trex_compressed.csv')
    else:
        sheet = pd.read_csv(spreadsheet_root + 'prime_trex_compressed_new.csv')
    ids = sheet['Patient_ID'].to_numpy()

    return np.unique(ids)

def get_patient_ids_by_visit(spreadsheet_root, max_val, mode='tr'):
    '''
    Returns unique patient IDs in given train/test split, where only patients have
    certain #s of visits are considered (max_val)
    '''
    spreadsheet_root = spreadsheet_root + '/spreadsheets/'
    if mode == 'tr':
        sheet = pd.read_csv(spreadsheet_root + 'prime_trex_compressed.csv')
    else:
        sheet = pd.read_csv(spreadsheet_root + 'prime_trex_compressed_new.csv')
    ids = sheet['Patient_ID'].to_numpy()
    unique_ids = np.unique(ids)
    new_ids = []
    for i in unique_ids:
        subsheet = sheet[sheet['Patient_ID'] == i]
        max_visit = subsheet['Visit'].max()
        # if patient has at least 'max_val' number of visits, include it
        if max_visit >= max_val:
            new_ids.append(i)
    return np.array(new_ids)