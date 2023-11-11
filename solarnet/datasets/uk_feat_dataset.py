import glob
import torch
import random
import pickle
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import Optional, List, Tuple
from torchvision.transforms import v2, GaussianBlur

from .utils import normalize
from .transforms import no_change, horizontal_flip, vertical_flip, colour_jitter

from solarnet.config import UK_1M_DATASET_PATH, UK_20K_v2_DATASET_PATH, UK_1M_FEATS_PATH
from solarnet.datasets import make_masks

def _get_subsample(all_paths, per_class_sample_size):
    pos = {name:array for name, array in all_paths.items() if name.endswith('-P.tif')}
    neg = {name:array for name, array in all_paths.items() if name.endswith('-N.tif')}
    pos_indices = torch.randperm(len(pos))[:per_class_sample_size]
    neg_indices = torch.randperm(len(neg))[:per_class_sample_size]
    pos_items, neg_items = list(pos.items()), list(neg.items())
    pos_sub = {pos_items[i][0]:pos_items[i][1] for i in pos_indices}
    neg_sub = {neg_items[i][0]:neg_items[i][1] for i in neg_indices}
    all_paths_new = {}
    all_paths_new.update(pos_sub)
    all_paths_new.update(neg_sub)
    return all_paths_new

def create_uk_feat_dataloaders(pkl_file, per_class_sample_size, validation1_frac, validation2_frac):
    
    with open(pkl_file, 'rb') as handle:
        all_paths = pickle.load(handle)
    all_paths = _get_subsample(all_paths, per_class_sample_size)
    
    train_dataset = UKFeatDataset(all_paths)
    len_dataset = len(train_dataset)
    train_mask, val1_mask, val2_mask = make_masks(len_dataset, validation1_frac, validation2_frac)
    train_dataset.add_mask(train_mask)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    val1_dataloader = DataLoader(UKFeatDataset(all_paths, mask=val1_mask),
                                 batch_size=64, shuffle=True, num_workers=8)
    val2_dataloader = DataLoader(UKFeatDataset(all_paths, mask=val2_mask),
                                 batch_size=64, shuffle=True, num_workers=8)
    
    print('Train iterations per epoch:', len(train_dataloader))
    print('Val1 iterations:', len(val1_dataloader))
    print('Val2 iterations:', len(val2_dataloader))
    
    return train_dataloader, val1_dataloader, val2_dataloader

class UKFeatDataset:

    def __init__(self, all_paths,
                 mask: Optional[List[bool]] = None) -> None:
        self.all_paths = all_paths
        if mask is not None:
            self.add_mask(mask)
    
    def add_mask(self, mask: List[bool]) -> None:
        self.all_paths = [(name, array) for include, (name, array) in zip(mask, self.all_paths.items()) if include]
        self.positive_paths = [(name, array) for name, array in self.all_paths if name.endswith('-P.tif')]
        self.negative_paths = [(name, array) for name, array in self.all_paths if name.endswith('-N.tif')]
        print(f'Positive: {len(self.positive_paths)} -- Negative: {len(self.negative_paths)}')
    
    def __len__(self) -> int:
        return len(self.all_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name, array = self.all_paths[index]
        x = array
        y = torch.tensor(1.0).long() if name.endswith('-P.tif') else torch.tensor(0.0).long() 
        return x, y, name
