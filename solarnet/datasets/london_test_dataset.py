import cv2
import glob
import torch
import pickle
import numpy as np
from .utils import normalize
from torch.utils.data import Dataset
from torchvision.transforms import v2, GaussianBlur

class TestDataset(Dataset):
    
    def __init__(self, path, MEAN = [0.485, 0.456, 0.406], STD=[0.229, 0.224, 0.225]):
        self.p_img_dirs = sorted(glob.glob(f'{path}/positive/*.jpg'))
        self.n_img_dirs = sorted(glob.glob(f'{path}/negative/*.jpg'))
        self.MEAN = MEAN
        self.STD = STD

    def __len__(self):
        return len(self.p_img_dirs) + len(self.n_img_dirs)

    def __getitem__(self, idx):
        if idx < len(self.n_img_dirs): # empty
            x = cv2.imread(self.n_img_dirs[idx])[:, :, ::-1]
            x = x.transpose((2, 0, 1))
            # x = self._transform_images(x)
            x = normalize(x, MEAN=self.MEAN, STD=self.STD)
            x = torch.from_numpy(x)
            return x.type(torch.float32), torch.tensor(0.0), self.n_img_dirs[idx]
        else: # solar
            idx = idx % len(self.p_img_dirs)
            x = cv2.imread(self.p_img_dirs[idx])[:, :, ::-1]
            x = x.transpose((2, 0, 1))
            # x = self._transform_images(x)
            x = normalize(x, MEAN=self.MEAN, STD=self.STD)
            x = torch.from_numpy(x)
            return x.type(torch.float32), torch.tensor(1.0), self.p_img_dirs[idx]
    
    def _transform_images(self, image: np.ndarray) -> np.ndarray:
        jitter = v2.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        g_blur = GaussianBlur(kernel_size=3, sigma=0.5)
        tens = torch.from_numpy(image)
        # adding jitter
        tens = jitter(tens)
        # adding guassian blur
        # tens = g_blur(tens)
        npy = tens.numpy()
        return npy
    
    
class London300Feats(Dataset):
    def __init__(self, pkl_file):
        
        with open(pkl_file, 'rb') as handle:
            all_paths = pickle.load(handle)
        self.all_paths = list(all_paths.items())
        
    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        name, array = self.all_paths[idx]
        x = array
        y = torch.tensor(1.0).long() if 'positive' in name else torch.tensor(0.0).long() 
        return x, y, name