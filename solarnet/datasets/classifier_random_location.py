import torch
import random
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from torchvision.transforms import v2, GaussianBlur

from typing import Optional, List, Tuple

from .utils import normalize
from .transforms import no_change, horizontal_flip, vertical_flip, colour_jitter

IMAGE_SIZES = {
    'Modesto': (5000, 5000),
    'Fresno': (5000, 5000),
    'Oxnard': (4000, 6000),
    'Stockton': (5000, 5000)
}

class ClassifierRandomLocationDataset:

    def __init__(self,
                 data_folder: Path=Path('data'),
                 normalize: bool = True, 
                 transform_images: bool = False,
                 mask: Optional[List[bool]] = None) -> None:

        self.normalize = normalize
        self.transform_images = transform_images
        self.data_folder = data_folder
        
        self.counter = 0
        self.output_dict = self.read_centroids()
        self.city_name = []
        self.city_name_loc = []
        for city in self.output_dict.keys():
            for name in self.output_dict[city].keys():
                self.city_name.append((city, name))
                
        if mask is not None:
            self.add_mask(mask)
        
    def add_mask(self, mask: List[bool]) -> None:
        """Add a mask to the data
        """
        self.city_name = [x for include, x in zip(mask, self.city_name) if include]
        for city, name in self.city_name:
            for loc in self.output_dict[city][name]:
                self.city_name_loc.append((city, name, loc))
            
            
    def read_centroids(self) -> defaultdict:

        metadata = pd.read_csv(self.data_folder / 'metadata/polygonDataExceptVertices.csv',
                               usecols=['city', 'image_name', 'centroid_latitude_pixels',
                                        'centroid_longitude_pixels'])
        org_len = len(metadata)
        metadata = metadata.dropna()
        print(f'Dropped {org_len - len(metadata)} rows due to NaN values')

        # for each image, we want to know where the solar panel centroids are
        output_dict: defaultdict = defaultdict(lambda: defaultdict(set))

        for idx, row in metadata.iterrows():
            output_dict[row.city][row.image_name].add((
                row.centroid_latitude_pixels, row.centroid_longitude_pixels
            ))
        return output_dict
    
    def __len__(self) -> int:
        return 2 * len(self.city_name_loc)

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

    # todo: optimize this function
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if index % 2 == 0:
            # solar
            item = self.city_name_loc[torch.randint(0, len(self.city_name_loc), (1,))]
            org_file = rasterio.open(self.data_folder / f"{item[0]}/{item[1]}.tif").read()   
            if org_file.shape != (3, IMAGE_SIZES[item[0]][0], IMAGE_SIZES[item[0]][1]):
                return self.__getitem__(index)
            center_h, center_w = int(item[2][0]), int(item[2][1])
            h_begin = torch.randint(max(0, center_h-224), min(IMAGE_SIZES[item[0]][0]-224, center_h), (1,))
            w_begin = torch.randint(max(0, center_w-224), min(IMAGE_SIZES[item[0]][1]-224, center_w), (1,))
            x = org_file[:, h_begin:h_begin+224, w_begin:w_begin+224]
            if self.transform_images: x = self._transform_images(x)
            if self.normalize: x = normalize(x)
            return torch.as_tensor(x.copy()).float(), torch.tensor(1.0)
                     
        else:
            # empty
            item = self.city_name[torch.randint(0, len(self.city_name), (1,))]
            org_file = rasterio.open(self.data_folder / f"{item[0]}/{item[1]}.tif").read()   
            if org_file.shape != (3, IMAGE_SIZES[item[0]][0], IMAGE_SIZES[item[0]][1]):
                return self.__getitem__(index)
            while True:
                img_size = IMAGE_SIZES[item[0]]
                rand_h = torch.randint(112, img_size[0]-112, (1,))
                rand_w = torch.randint(112, img_size[1]-112, (1,))
                brk = True
                for h, w in self.output_dict[item[0]][item[1]]:
                    if h < rand_h + 112 and h >= rand_h - 112 and w < rand_w + 112 and w >= rand_w - 112:
                        brk = False
                        break
                if brk: break
            x = org_file[:, rand_h-112:rand_h+112, rand_w-112:rand_w+112]
            if self.transform_images: x = self._transform_images(x)
            if self.normalize: x = normalize(x)
            return torch.as_tensor(x.copy()).float(), torch.tensor(0.0)
