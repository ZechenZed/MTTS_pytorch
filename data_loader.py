import torch
import numpy as np
from torch.utils.data import Dataset


class V4V_Dataset(Dataset):
    def __init__(self, data_folder_path, data_type, image_type, BP_type, filter_type):
        print(f'Testing on {data_type} set')
        frames = np.load(f'{data_folder_path}{data_type}_frames_{image_type}.npy').astype(np.float32)
        BP = np.load(f'{data_folder_path}{data_type}_BP_{BP_type}_{filter_type}.npy').astype(np.float32)
        self.X = torch.from_numpy(frames)
        self.Y = torch.from_numpy(BP)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
