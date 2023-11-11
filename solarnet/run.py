import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from solarnet.datasets import TestDataset
from solarnet.preprocessing import MaskMaker, ImageSplitter
from solarnet.models import Classifier, Segmenter, train_classifier, train_segmenter, DinoClassifier
from solarnet.datasets import (ClassifierRandomLocationDataset, ClassifierDataset, SegmenterDataset,
                               make_masks, make_masks_torch, UKDataset, UKDatasetFull, UKFeatDataset, create_uk_feat_dataloaders)

from solarnet.config import (UK_1M_DATASET_PATH,
                             UK_20K_v2_DATASET_PATH,
                             UK_1M_FEATS_PATH,
                             LONDON_300_PATH)

def init_exp(exp_dir):
    # instance directory
    instances = [int(x) for x in next(os.walk(exp_dir))[1]]
    if len(instances) == 0:
        instance_id = 1
    else:
        instance_id = max(instances) + 1
    instance_dir = exp_dir / f'{instance_id}'
    checkpoints_dir = instance_dir / 'checkpoints'
    if not instance_dir.exists(): instance_dir.mkdir()
    if not checkpoints_dir.exists(): checkpoints_dir.mkdir()
    return instance_dir

class RunTask:

    @staticmethod
    def make_masks(data_folder='data'):
        """Saves masks for each .tif image in the raw dataset. Masks are saved
        in  <org_folder>_mask/<org_filename>.npy where <org_folder> should be the
        city name, as defined in `data/README.md`.

        Parameters
        ----------
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        """
        mask_maker = MaskMaker(data_folder=Path(data_folder))
        mask_maker.process()

    @staticmethod
    def split_images(data_folder='data', imsize=224, empty_ratio=2):
        """Generates images (and their corresponding masks) of height = width = imsize
        for input into the models.

        Parameters
        ----------
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        imsize: int, default: 224
            The size of the images to be generated
        empty_ratio: int, default: 2
            The ratio of images without solar panels to images with solar panels.
            Because images without solar panels are randomly sampled with limited
            patience, having this number slightly > 1 yields a roughly 1:1 ratio.
        """
        splitter = ImageSplitter(data_folder=Path(data_folder))
        splitter.process(imsize=imsize, empty_ratio=empty_ratio)

    @staticmethod
    def extract_dino_features(data_folder='data', mode='small',
                              device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),):
        
        model = DinoClassifier(mode=mode, layers=1, stacked_layers=2)
        model = model.to(device)
        
        # dataset = UKDatasetFull(data_folder=Path(UK_1M_DATASET_PATH), train_mode=False, transform_images=False)
        # dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
        
        dataset = TestDataset(LONDON_300_PATH, MEAN=[0.5, 0.5, 0.5], STD=[0.5, 0.5, 0.5])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        len_dataset = len(dataset)
        
        with torch.no_grad():
            model.eval()
            features = {}
            for batch_num, (x, y, names) in enumerate(dataloader):
                print(f'Processing batch {batch_num}')
                x = x.to(device)
                y = y.to(device)
                feats = model.get_dino_features(x)
                for i in range(x.shape[0]):
                    features[names[i]] = feats[i].cpu().numpy()
                if ((batch_num + 1) % 200 == 0) or (batch_num + 1 == len(dataloader)):
                    np.save(f'./data/temp/{batch_num}', features, allow_pickle=True)
                    print(f'-------------- Saved in file ------------------')
                    features = {}

# train_mlp1
# train_mlp2   

    @staticmethod
    def train_mlp2(max_epochs=100, warmup=2, patience=5, val_size=0.1,
                   test_size=0.1, data_folder='data', mode='small', train_mode='freeze',
                   device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                   exp_dir=None, exp_name=None):
        
        instance_dir = init_exp(Path(exp_dir))
        writer = SummaryWriter(f'runs/{exp_name}')
        train_dl, val1_dl, val2_dl = create_uk_feat_dataloaders(UK_1M_FEATS_PATH, 10_000, 0.19, 0.01)
        
        model = nn.Sequential(
            nn.Linear(3072, 300),
            nn.ReLU(),
            nn.Linear(300, 2)
        )
        model = model.to(device)
        learning_rate = 1e-5
    
        train_classifier(model, train_dl, val1_dl, max_epochs=max_epochs,
                         warmup=warmup, patience=patience, device=device,
                         instance_dir=instance_dir, writer=writer, learning_rate=learning_rate, only_feats=True)
 

    @staticmethod
    def train_classifier(max_epochs=100, warmup=2, patience=5, val_size=0.1,
                         test_size=0.1, data_folder='data', mode='small', train_mode='freeze',
                         device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                         exp_dir=None, exp_name=None):
        """Train the classifier

        Parameters
        ----------
        max_epochs: int, default: 100
            The maximum number of epochs to train for
        warmup: int, default: 2
            The number of epochs for which only the final layers (not from the ResNet base)
            should be trained
        patience: int, default: 5
            The number of epochs to keep training without an improvement in performance on the
            validation set before early stopping
        val_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the validation set
        test_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the test set
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        device: torch.device, default: cuda if available, else cpu
            The device to train the models on
        """
        data_folder = Path(data_folder)
        instance_dir = init_exp(Path(exp_dir))
        writer = SummaryWriter(f'runs/{exp_name}')

        model = DinoClassifier(mode=mode, layers=1, stacked_layers=2)
        
        if train_mode == 'freeze':
            learning_rate = 1e-5
            for name, p in model.named_parameters():
                if 'linear_head' not in name:
                    p.requires_grad = False 
        elif train_mode == 'finetune':
            learning_rate = 1e-5
        else:
            raise Exception(f'train mode "{train_mode}" not defined.')

        if device.type != 'cpu': model = model.cuda()
        
        # processed_folder = data_folder / 'processed'
        # dataset = ClassifierRandomLocationDataset(transform_images=False)
        # len_dataset = len(dataset.city_name)
        
        # dataset = UKDataset(data_folder=Path('./data/labeled_data_from_colab'), transform_images=False)
        # len_dataset = len(dataset)

        dataset = UKDatasetFull(data_folder=Path(UK_20K_v2_DATASET_PATH), transform_images=False)
        len_dataset = len(dataset)

        # make a train and val set
        train_mask, sub_val_mask, full_val_mask = make_masks(len_dataset, 0.19, 0.01)
        # train_mask, sub_val_mask, full_val_mask = make_masks(len_dataset, 0.15, 0.05)
        dataset.add_mask(train_mask)
        
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
        
        sub_val_dataloader = DataLoader(UKDatasetFull(data_folder=Path(UK_20K_v2_DATASET_PATH), 
                                                      mask=sub_val_mask, transform_images=False, train_mode=False),
                                                      batch_size=64, shuffle=True, num_workers=8)
        
        full_val_dataloader = DataLoader(UKDatasetFull(data_folder=Path(UK_20K_v2_DATASET_PATH),
                                                       mask=full_val_mask, transform_images=False, train_mode=False),
                                                       batch_size=64, shuffle=True, num_workers=8)
        
        print('Train iterations per epoch:', len(train_dataloader))
        print('Subval iterations:', len(sub_val_dataloader))
        print('Fullval iterations:', len(full_val_dataloader))

        train_classifier(model, train_dataloader, sub_val_dataloader, max_epochs=max_epochs,
                         warmup=warmup, patience=patience, device=device, instance_dir=instance_dir, writer=writer, learning_rate=learning_rate)

        # save predictions for analysis
        # print("Generating test results")
        # preds, true = [], []
        # with torch.no_grad():
        #     for test_x, test_y in tqdm(test_dataloader):
        #         test_x = test_x.to(device)
        #         test_y = test_y.to(device)
        #         test_preds = model(test_x)
        #         preds.append(test_preds.squeeze(1).cpu().numpy())
        #         true.append(test_y.cpu().numpy())
        # np.save(savedir / 'classifier_preds.npy', np.concatenate(preds))
        # np.save(savedir / 'classifier_true.npy', np.concatenate(true))

    @staticmethod
    def train_segmenter(max_epochs=100, val_size=0.1, test_size=0.1, warmup=2,
                        patience=5, data_folder='data', use_classifier=True,
                        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        """Train the segmentation model

        Parameters
        ----------
        max_epochs: int, default: 100
            The maximum number of epochs to train for
        warmup: int, default: 2
            The number of epochs for which only the final layers (not from the ResNet base)
            should be trained
        patience: int, default: 5
            The number of epochs to keep training without an improvement in performance on the
            validation set before early stopping
        val_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the validation set
        test_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the test set
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        use_classifier: boolean, default: True
            Whether to use the pretrained classifier (saved in data/models/classifier.model by the
            train_classifier step) as the weights for the downsampling step of the segmentation
            model
        device: torch.device, default: cuda if available, else cpu
            The device to train the models on
        """
        data_folder = Path(data_folder)
        model = Segmenter()
        if device.type != 'cpu': model = model.cuda()

        model_dir = data_folder / 'models'
        if use_classifier:
            classifier_sd = torch.load(model_dir / 'classifier.model')
            model.load_base(classifier_sd)
        processed_folder = data_folder / 'processed'
        dataset = SegmenterDataset(processed_folder=processed_folder)
        train_mask, val_mask, test_mask = make_masks(len(dataset), val_size, test_size)

        dataset.add_mask(train_mask)
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(SegmenterDataset(mask=val_mask,
                                                     processed_folder=processed_folder,
                                                     transform_images=False),
                                    batch_size=64, shuffle=True)
        test_dataloader = DataLoader(SegmenterDataset(mask=test_mask,
                                                      processed_folder=processed_folder,
                                                      transform_images=False),
                                     batch_size=64)

        train_segmenter(model, train_dataloader, val_dataloader, max_epochs=max_epochs,
                        warmup=warmup, patience=patience)

        if not model_dir.exists(): model_dir.mkdir()
        torch.save(model.state_dict(), model_dir / 'segmenter.model')

        print("Generating test results")
        images, preds, true = [], [], []
        with torch.no_grad():
            for test_x, test_y in tqdm(test_dataloader):
                test_preds = model(test_x)
                images.append(test_x.cpu().numpy())
                preds.append(test_preds.squeeze(1).cpu().numpy())
                true.append(test_y.cpu().numpy())

        np.save(model_dir / 'segmenter_images.npy', np.concatenate(images))
        np.save(model_dir / 'segmenter_preds.npy', np.concatenate(preds))
        np.save(model_dir / 'segmenter_true.npy', np.concatenate(true))

    def train_both(self, c_max_epochs=100, c_warmup=2, c_patience=5, c_val_size=0.1,
                   c_test_size=0.1, s_max_epochs=100, s_warmup=2, s_patience=5,
                   s_val_size=0.1, s_test_size=0.1, data_folder='data',
                   device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        """Train the classifier, and use it to train the segmentation model.
        """
        data_folder = Path(data_folder)
        self.train_classifier(max_epochs=c_max_epochs, val_size=c_val_size, test_size=c_test_size,
                              warmup=c_warmup, patience=c_patience, data_folder=data_folder,
                              device=device)
        self.train_segmenter(max_epochs=s_max_epochs, val_size=s_val_size, test_size=s_test_size,
                             warmup=s_warmup, patience=s_patience, use_classifier=True,
                             data_folder=data_folder, device=device)
