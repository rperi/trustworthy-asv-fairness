import torch
import numpy as np
import os
from torch.utils.data import Dataset
from collections import Counter
import pdb

class FairVoice_embeddings(Dataset):
    def __init__(self, data_root, bias_type='gender', valFlag=False):
        if not valFlag:
            self.data = np.load(os.path.join(data_root,'train_data.npy'))
            self.labels_spk = np.load(os.path.join(data_root,'train_labels.npy'))
            self.labels_bias = np.load(os.path.join(data_root,'train_labels_{}.npy'.format(bias_type)))
            counts = Counter(self.labels_bias)
            self.prob_bias = {}
            for idx, key in enumerate(list(counts.keys())):
                self.prob_bias[key] = self.labels_bias.shape[0]/counts[key]
            total = np.sum([self.prob_bias[key] for key in counts.keys()])
            for idx, key in enumerate(list(counts.keys())):
                self.prob_bias[key] = self.prob_bias[key]/total
        else:
            self.data = np.load(os.path.join(data_root, 'val_data.npy'))
            self.labels_spk = np.load(os.path.join(data_root, 'val_labels.npy'))
            self.labels_bias = np.load(os.path.join(data_root,'val_labels_{}.npy'.format(bias_type)))
        self.feat_dim = self.data.shape[1]
        self.num_spk = np.unique(self.labels_spk).shape[0] #np.unique(self.labels).shape[0]
        self.num_bias = np.unique(self.labels_bias).shape[0] #np.unique(self.labels).shape[0]
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx,:], self.labels_spk[idx], self.labels_bias[idx]

class FairVoice_embeddings_test(Dataset):
    def __init__(self, data_path):
        self.data = torch.from_numpy(np.load(data_path)).type(torch.FloatTensor)
        self.feat_dim = self.data.shape[1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx,:]
