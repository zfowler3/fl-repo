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