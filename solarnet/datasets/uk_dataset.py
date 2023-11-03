import glob
import torch
import random
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, List, Tuple
from torchvision.transforms import v2, GaussianBlur

from .utils import normalize
from .transforms import no_change, horizontal_flip, vertical_flip, colour_jitter

IMAGE_SIZES = {
    'Modesto': (5000, 5000),
    'Fresno': (5000, 5000),
    'Oxnard': (4000, 6000),
    'Stockton': (5000, 5000)
}

class UKDatasetFull:

    def __init__(self,
                 data_folder: Path=Path('data'),
                 normalize: bool = True, 
                 transform_images: bool = False,
                 mask: Optional[List[bool]] = None) -> None:

        self.normalize = normalize
        self.transform_images = transform_images
        self.data_folder = data_folder
        
        self.all_paths = sorted(glob.glob(f'{data_folder}/*'))
        if mask is not None:
            self.add_mask(mask)
    
    def add_mask(self, mask: List[bool]) -> None:
        self.all_paths = [x for include, x in zip(mask, self.all_paths) if include]
        self.positive_paths = [x for x in self.all_paths if x.endswith('-P.tif')]
        self.negative_paths = [x for x in self.all_paths if x.endswith('-N.tif')]
        
        print(len(self.positive_paths))
        print(len(self.negative_paths))
    
    def __len__(self) -> int:
        return len(self.all_paths)

    # def _transform_images(self, image: np.ndarray) -> np.ndarray:
    #     transforms = [
    #         no_change,
    #         horizontal_flip,
    #         vertical_flip,
    #         colour_jitter,
    #     ]
    #     chosen_function = random.choice(transforms)
    #     return chosen_function(image)
    
    def _transform_images(self, image: np.ndarray) -> np.ndarray:
        
        jitter = v2.ColorJitter(brightness=.5, contrast=0.5, saturation=0.5, hue=.3)
        g_blur = GaussianBlur(kernel_size=3, sigma=0.5)
        tens = torch.from_numpy(image)

        # adding jitter
        if torch.rand((1, )) > 0.5:
            tens = jitter(tens)

        # adding guassian blur
        if torch.rand((1, )) > 0.5:
            tens = g_blur(tens)

        npy = tens.numpy()
        
        # fig, axes = plt.subplots(1, 2)
        # axes[0].imshow(image.transpose((1, 2, 0)))
        # axes[1].imshow(npy.transpose((1, 2, 0)))
        # axes[0].axis('off')
        # axes[1].axis('off')
        # plt.savefig(f'tmp_plots/{self.counter}.png')
        # plt.close()
        # self.counter += 1
        
        return npy

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.all_paths[index]
        x = rasterio.open(path).read()
        y = torch.tensor(1.0).long() if path.endswith('-P.tif') else torch.tensor(0.0).long() 
        if self.transform_images: x = self._transform_images(x)
        if self.normalize: x = normalize(x, MEAN=[0.5, 0.5, 0.5], STD=[0.5, 0.5, 0.5])
        return torch.as_tensor(x.copy()).float(), y


class UKDataset:

    def __init__(self,
                 data_folder: Path=Path('data'),
                 normalize: bool = True, 
                 transform_images: bool = False,
                 mask: Optional[List[bool]] = None) -> None:

        self.normalize = normalize
        self.transform_images = transform_images
        self.data_folder = data_folder
        
        self.all_paths = sorted(glob.glob(f'{data_folder}/*'))
        if mask is not None:
            self.add_mask(mask)
    
    def add_mask(self, mask: List[bool]) -> None:
        self.all_paths = [x for include, x in zip(mask, self.all_paths) if include]
        self.positive_paths = [x for x in self.all_paths if x.endswith('-P.tif')]
        self.negative_paths = [x for x in self.all_paths if x.endswith('-N.tif')]
    
    def __len__(self) -> int:
        return len(self.all_paths)

    # def _transform_images(self, image: np.ndarray) -> np.ndarray:
    #     transforms = [
    #         no_change,
    #         horizontal_flip,
    #         vertical_flip,
    #         colour_jitter,
    #     ]
    #     chosen_function = random.choice(transforms)
    #     return chosen_function(image)
    
    def _transform_images(self, image: np.ndarray) -> np.ndarray:
        
        jitter = v2.ColorJitter(brightness=.5, contrast=0.5, saturation=0.5, hue=.3)
        g_blur = GaussianBlur(kernel_size=3, sigma=0.5)
        tens = torch.from_numpy(image)

        # adding jitter
        if torch.rand((1, )) > 0.5:
            tens = jitter(tens)

        # adding guassian blur
        if torch.rand((1, )) > 0.5:
            tens = g_blur(tens)

        npy = tens.numpy()
        
        # fig, axes = plt.subplots(1, 2)
        # axes[0].imshow(image.transpose((1, 2, 0)))
        # axes[1].imshow(npy.transpose((1, 2, 0)))
        # axes[0].axis('off')
        # axes[1].axis('off')
        # plt.savefig(f'tmp_plots/{self.counter}.png')
        # plt.close()
        # self.counter += 1
        
        return npy

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.all_paths[index]
        x = rasterio.open(path).read()
        y = torch.tensor(1.0).long() if path.endswith('-P.tif') else torch.tensor(0.0).long() 
        if self.transform_images: x = self._transform_images(x)
        if self.normalize: x = normalize(x, MEAN=[0.5, 0.5, 0.5], STD=[0.5, 0.5, 0.5])
        return torch.as_tensor(x.copy()).float(), y
