import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import cv2 as cv

class Olives_DiseaseDetection(data.Dataset):
    def __init__(self, root='img_directory/', transforms=None,
                 dataidxs=None, mode='tr', spreadsheet_root='/home/zoe/olives'):
        self.img_dir = root
        self.transform = transforms
        # read spreadsheet
        rroot = spreadsheet_root
        if mode == 'tr':
            rroot = rroot + '/prime_trex_compressed.csv'
        else:
            rroot = rroot + '/prime_trex_test_new.csv'
        self.x = pd.read_csv(rroot)
        self.targets = self.x['Label'].to_numpy()

        if dataidxs is not None:
            self.x = self.x.iloc[dataidxs].reset_index().iloc[:, 1:]
            self.targets = self.targets[dataidxs]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        path = self.img_dir + self.x.iloc[idx,0]
        image = Image.open(path)
        image = np.array(image, dtype=object)
        im = image.astype('uint8')
        if len(image.shape) == 3:
            im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        im = cv.resize(im, (128, 128), interpolation=cv.INTER_AREA)
        image = Image.fromarray(im)
        image = self.transform(image)
        label = self.targets[idx]
        return image, label, idx