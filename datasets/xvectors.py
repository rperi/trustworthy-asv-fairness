import torch
import numpy as np
import os
from torch.utils.data import Dataset

# Unlike FV dataset, this dataset does not use bias labels
class Xvectors(Dataset):
    def __init__(self, data_root, valFlag=False):
        if not valFlag:
            self.data = np.load(os.path.join(data_root,'train_data.npy'))
            self.labels = np.load(os.path.join(data_root,'train_labels.npy'))
            self.feat_dim = self.data.shape[1]
            self.num_spk = np.unique(self.labels).shape[0] #np.unique(self.labels).shape[0]
        else:
            self.data = np.load(os.path.join(data_root, 'val_data.npy'))
            self.labels = np.load(os.path.join(data_root, 'val_labels.npy'))


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx,:], self.labels[idx]

class Xvectors_test(Dataset):
    def __init__(self, data_path):
        self.data = torch.from_numpy(np.load(data_path)).type(torch.FloatTensor)
        self.feat_dim = self.data.shape[1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx,:]
