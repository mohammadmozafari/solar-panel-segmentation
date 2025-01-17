import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from solarnet.datasets import denormalize
import cv2
from typing import Any, List, Tuple
from solarnet.utils import report_validation_results
from solarnet.datasets import TestDataset, London300Feats

from solarnet.config import (LONDON_300_PATH,
                             LONDON_300_FEATS_PATH)

def train_classifier(model: torch.nn.Module,
                     train_dataloader: DataLoader,
                     sub_val_dataloader: DataLoader,
                     warmup: int = 2,
                     patience: int = 5,
                     max_epochs: int = 100,
                     device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                     instance_dir=None,
                     writer=None,
                     learning_rate=1e-3,
                     only_feats=False) -> None:
    """Train the classifier

    Parameters
    ----------s
    model
        The classifier to be trained
    train_dataloader:
        An iterator which returns batches of training images and labels from the
        training dataset
    sub_val_dataloader:
        An iterator which returns batches of training images and labels from the
        validation dataset
    warmup: int, default: 2
        The number of epochs for which only the final layers (not from the ResNet base)
        should be trained
    patience: int, default: 5
        The number of epochs to keep training without an improvement in performance on the
        validation set before early stopping
    max_epochs: int, default: 100
        The maximum number of epochs to train for
    """
    
    lr = learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, min_lr=1e-7, factor=0.5)
    
    iteration = 0
    best_f1 = 0
    t_losses, t_true, t_pred = [], [], []
    for epoch in range(max_epochs):

        model.train()
        running_losses = []
    
        for x, y, _ in train_dataloader:
            
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()        
            logits = model(x)

            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            t_losses.append(loss.item())
            running_losses.append(loss.item())

            pred = F.softmax(logits, -1)[:, 1]
            t_true.append(y.detach().cpu().numpy())
            t_pred.append(pred.detach().cpu().numpy())
            
            if (iteration + 1) % 10 == 0:
                t_loss = sum(running_losses) / len(running_losses)
                running_losses = []
                print(f'[{epoch}, {iteration}] -- Loss: {t_loss}')
                writer.add_scalar('HighFreqLoss', t_loss, iteration)
                lr_scheduler.step(t_loss)
                
            if (iteration + 1) % 200 == 0:
                v_losses, v_true, v_pred = validate_on_sub_val(model, sub_val_dataloader, device)
                
                if only_feats:
                    prec, rec, f1, acc = validate_on_london_only_feats(model, device, 0.5, writer, iteration)
                else:
                    prec, rec, f1, acc = validate_on_london(model, device, 0.5, writer, iteration)
                
                best_threshold = report_validation_results(t_losses, t_true, t_pred, v_losses, v_true, v_pred, writer, iteration)
                
                if only_feats:
                    _ = validate_on_london_only_feats(model, device, best_threshold, writer, iteration)
                else:
                    _ = validate_on_london(model, device, best_threshold, writer, iteration)
                
                t_losses, t_true, t_pred = [], [], []
                
                if f1 > best_f1:
                    best_f1 = f1
                    savedir = instance_dir / 'checkpoints'
                    if not savedir.exists(): savedir.mkdir()
                    torch.save(model.state_dict(), savedir / f'it{iteration}.model')
            
            iteration += 1

def train_classifier_single_output(model: torch.nn.Module,
                                   train_dataloader: DataLoader,
                                   sub_val_dataloader: DataLoader,
                                   warmup: int = 2,
                                   patience: int = 5,
                                   max_epochs: int = 100,
                                   device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                                   instance_dir=None,
                                   writer=None,
                                   learning_rate=1e-3,
                                   only_feats=False) -> None:
    
    lr = learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, min_lr=1e-7, factor=0.5)
    
    iteration = 0
    best_f1 = 0
    t_losses, t_true, t_pred = [], [], []
    for epoch in range(max_epochs):

        model.train()
        running_losses = []
    
        for x, y, _ in train_dataloader:
            
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()        
            pos_prob = model(x)

            pred = pos_prob[:, 0]
            loss = F.binary_cross_entropy(pred, y.float())
            loss.backward()
            optimizer.step()
            t_losses.append(loss.item())
            running_losses.append(loss.item())
            
            t_true.append(y.detach().cpu().numpy())
            t_pred.append(pred.detach().cpu().numpy())
            
            if (iteration + 1) % 10 == 0:
                t_loss = sum(running_losses) / len(running_losses)
                running_losses = []
                print(f'[{epoch}, {iteration}] -- Loss: {t_loss}')
                writer.add_scalar('HighFreqLoss', t_loss, iteration)
                lr_scheduler.step(t_loss)
                
            if (iteration + 1) % 200 == 0:
                v_losses, v_true, v_pred = validate_on_sub_val(model, sub_val_dataloader, device)
                
                if only_feats:
                    prec, rec, f1, acc = validate_on_london_only_feats(model, device, 0.5, writer, iteration)
                else:
                    prec, rec, f1, acc = validate_on_london(model, device, 0.5, writer, iteration)
                
                best_threshold = report_validation_results(t_losses, t_true, t_pred, v_losses, v_true, v_pred, writer, iteration)
                
                if only_feats:
                    _ = validate_on_london_only_feats(model, device, best_threshold, writer, iteration)
                else:
                    _ = validate_on_london(model, device, best_threshold, writer, iteration)
                
                t_losses, t_true, t_pred = [], [], []
                
                if f1 > best_f1:
                    best_f1 = f1
                    savedir = instance_dir / 'checkpoints'
                    if not savedir.exists(): savedir.mkdir()
                    torch.save(model.state_dict(), savedir / f'it{iteration}.model')
            
            iteration += 1

def train_segmenter(model: torch.nn.Module,
                    train_dataloader: DataLoader,
                    val_dataloader: DataLoader,
                    warmup: int = 2,
                    patience: int = 5,
                    max_epochs: int = 100) -> None:
    """Train the segmentation model

    Parameters
    ----------
    model
        The segmentation model to be trained
    train_dataloader:
        An iterator which returns batches of training images and masks from the
        training dataset
    val_dataloader:
        An iterator which returns batches of training images and masks from the
        validation dataset
    warmup: int, default: 2
        The number of epochs for which only the upsampling layers (not trained by the classifier)
        should be trained
    patience: int, default: 5
        The number of epochs to keep training without an improvement in performance on the
        validation set before early stopping
    max_epochs: int, default: 100
        The maximum number of epochs to train for
    """
    best_state_dict = model.state_dict()
    best_loss = 1
    patience_counter = 0
    for i in range(max_epochs):
        if i <= warmup:
            # we start by 'warming up' the final layers of the model
            optimizer = torch.optim.Adam([pam for name, pam in
                                          model.named_parameters() if 'pretrained' not in name])
        else:
            optimizer = torch.optim.Adam(model.parameters())

        train_data, val_data = _train_segmenter_epoch(model, optimizer, train_dataloader,
                                                      val_dataloader)
        if np.mean(val_data) < best_loss:
            best_loss = np.mean(val_data)
            patience_counter = 0
            best_state_dict = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter == patience:
                print("Early stopping!")
                model.load_state_dict(best_state_dict)
                return None 

def _train_segmenter_epoch(model: torch.nn.Module,
                           optimizer: Optimizer,
                           train_dataloader: DataLoader,
                           val_dataloader: DataLoader
                           ) -> Tuple[List[Any], List[Any]]:
    t_losses, v_losses = [], []
    model.train()
    for x, y in tqdm(train_dataloader):
        optimizer.zero_grad()
        preds = model(x)

        loss = F.binary_cross_entropy(preds, y.unsqueeze(1))
        loss.backward()
        optimizer.step()

        t_losses.append(loss.item())

    with torch.no_grad():
        model.eval()
        for val_x, val_y in tqdm(val_dataloader):
            val_preds = model(val_x)
            val_loss = F.binary_cross_entropy(val_preds, val_y.unsqueeze(1))
            v_losses.append(val_loss.item())
    print(f'Train loss: {np.mean(t_losses)}, Val loss: {np.mean(v_losses)}')

    return t_losses, v_losses

def validate_on_london(model, device, thresh, writer, iteration):

    classifier = model

    ds = TestDataset(LONDON_300_PATH, MEAN=[0.5, 0.5, 0.5], STD=[0.5, 0.5, 0.5])
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    classifier.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for i, x in enumerate(dl):
            images = x[0].to(device)
            label = x[1].to(device)
            names = x[2]
            logit = classifier(images)
            pred = F.softmax(logit, -1)[:, 1]
            
            # print(names, "pred:", pred, "|", "label:", label)
            original_image = denormalize(images.cpu().numpy())
            pos_neg = f'{"P" if pred > thresh else "N"}'
            suffix = f'{pos_neg+"P" if label > 0.5 else pos_neg+"N"}'
            
            if pred >= 0.5 and label == 1: tp += 1
            elif pred >= 0.5 and label == 0: fp += 1
            elif pred < 0.5 and label == 1: fn += 1
            elif pred < 0.5 and label == 0: tn += 1
            else: raise Exception('Whatt ???')
            
            # img_name = save_path / f'{names[0].split("/")[-1].split(".")[0]}-{suffix}.png'
            # original_image = original_image[0, :, :, :].transpose((1, 2, 0))
            # cv2.imwrite(str(img_name), original_image[:, :, ::-1])

        print(f'TP: {tp} - TN: {tn}, FP: {fp} - FN: {fn}')
        
        # Precision attempts to answer the following question:
        # What proportion of positive identifications was actually correct?
        precision = np.NAN if tp + fp == 0 else tp/(tp+fp)
        print("Precision:", precision)

        # Recall attempts to answer the following question:
        # What proportion of actual positives was identified correctly?
        recall = np.NAN if tp + fn == 0 else tp/(tp+fn)
        print("Recall:", recall)

        # F1-score: a combo of precision and recall
        f1 = np.NAN if 2*tp + fp + fn == 0 else 2*tp/(2*tp + fp + fn)
        print("F1-score:",f1)
        
        accuracy = np.NAN if tp+fp+tn+fn == 0 else (tp+tn)/(tp+fp+tn+fn)
        print("Accuracy:",accuracy)
        
    writer.add_scalar('LondonTestSet/Precision', precision, iteration)
    writer.add_scalar('LondonTestSet/Recall', recall, iteration)
    writer.add_scalar('LondonTestSet/F1', f1, iteration)
    writer.add_scalar('LondonTestSet/Accuracy', accuracy, iteration)
    return precision, recall, f1, accuracy

def validate_on_london_only_feats(model, device, thresh, writer, iteration):

    classifier = model
    ds = London300Feats(LONDON_300_FEATS_PATH)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    classifier.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for i, x in enumerate(dl):
            images = x[0].to(device)
            label = x[1].to(device)
            names = x[2]
            logit = classifier(images)
            pred = F.softmax(logit, -1)[:, 1]
    
            if pred >= 0.5 and label == 1: tp += 1
            elif pred >= 0.5 and label == 0: fp += 1
            elif pred < 0.5 and label == 1: fn += 1
            elif pred < 0.5 and label == 0: tn += 1
            else: raise Exception('Whatt ???')

        print(f'TP: {tp} - TN: {tn}, FP: {fp} - FN: {fn}')
        
        # Precision attempts to answer the following question:
        # What proportion of positive identifications was actually correct?
        precision = np.NAN if tp + fp == 0 else tp/(tp+fp)
        print("Precision:", precision)

        # Recall attempts to answer the following question:
        # What proportion of actual positives was identified correctly?
        recall = np.NAN if tp + fn == 0 else tp/(tp+fn)
        print("Recall:", recall)

        # F1-score: a combo of precision and recall
        f1 = np.NAN if 2*tp + fp + fn == 0 else 2*tp/(2*tp + fp + fn)
        print("F1-score:",f1)
        
        accuracy = np.NAN if tp+fp+tn+fn == 0 else (tp+tn)/(tp+fp+tn+fn)
        print("Accuracy:",accuracy)
        
    writer.add_scalar('LondonTestSet/Precision', precision, iteration)
    writer.add_scalar('LondonTestSet/Recall', recall, iteration)
    writer.add_scalar('LondonTestSet/F1', f1, iteration)
    writer.add_scalar('LondonTestSet/Accuracy', accuracy, iteration)
    return precision, recall, f1, accuracy

def validate_on_sub_val(model, sub_val_dataloader, device):
    
    v_losses, v_true, v_pred = [], [], []
    
    with torch.no_grad():
        model.eval()
        num = 1
        for val_x, val_y, _ in tqdm(sub_val_dataloader):
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            
            val_logits = model(val_x)
            val_loss = F.cross_entropy(val_logits, val_y)
            v_losses.append(val_loss.item())

            val_preds = F.softmax(val_logits, -1)[:, 1]

            v_true.append(val_y.detach().cpu().numpy())
            v_pred.append(val_preds.detach().cpu().numpy())
    
    v_true = np.concatenate(v_true)
    v_pred = np.concatenate(v_pred)
    
    return v_losses, v_true, v_pred