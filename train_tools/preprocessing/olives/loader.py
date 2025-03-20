import pandas as pd

def get_patient_id(spreadsheet_root, mode='tr'):
    spreadsheet_root = spreadsheet_root + '/spreadsheets/'
    if mode == 'tr':
        sheet = pd.read_csv(spreadsheet_root + 'prime_trex_compressed.csv')
    else:
        sheet = pd.read_csv(spreadsheet_root + 'prime_trex_compressed_new.csv')
