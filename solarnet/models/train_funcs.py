import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from solarnet.datasets import denormalize
import cv2
from typing import Any, List, Tuple
from solarnet.datasets import TestDataset

def train_dino_classifier(model: torch.nn.Module,
                     train_dataloader: DataLoader,
                     val_dataloader: DataLoader,
                     warmup: int = 2,
                     patience: int = 5,
                     max_epochs: int = 100,
                     device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                     instance_dir=None,
                     writer=None) -> None:
    """Train the classifier

    Parameters
    ----------
    model
        The classifier to be trained
    train_dataloader:
        An iterator which returns batches of training images and labels from the
        training dataset
    val_dataloader:
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

    best_state_dict = model.state_dict()
    best_val_auc_roc = 0.5
    patience_counter = 0
    
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, min_lr=1e-7)
    # lr = 1e-5
    # lr = 1e-6
    
    iteration = [0]
    for epoch in range(max_epochs):

        train_data, val_data = _train_classifier_epoch(model, optimizer, train_dataloader,
                                                       val_dataloader, device, epoch, iteration, writer=writer, lr_scheduler=lr_scheduler)
        savedir = instance_dir / 'checkpoints'
        if not savedir.exists(): savedir.mkdir()
        torch.save(model.state_dict(), savedir / f'e{epoch}.model')
        
        if val_data[1] > best_val_auc_roc:
            best_val_auc_roc = val_data[1]
            patience_counter = 0
            best_state_dict = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter == patience:
                print("Early stopping!")
                model.load_state_dict(best_state_dict)
                return None

def train_classifier(model: torch.nn.Module,
                     train_dataloader: DataLoader,
                     val_dataloader: DataLoader,
                     warmup: int = 2,
                     patience: int = 5,
                     max_epochs: int = 100,
                     device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                     instance_dir=None) -> None:
    """Train the classifier

    Parameters
    ----------
    model
        The classifier to be trained
    train_dataloader:
        An iterator which returns batches of training images and labels from the
        training dataset
    val_dataloader:
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

    best_state_dict = model.state_dict()
    best_val_auc_roc = 0.5
    patience_counter = 0
    lr = 1e-5
    # lr = 1e-6
    
    for i in range(max_epochs):
        if i <= warmup:
            # we start by finetuning the model
            optimizer = torch.optim.Adam([pam for name, pam in
                                          model.named_parameters() if 'classifier' in name])
        else:
            # then, we train the whole thing
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_data, val_data = _train_classifier_epoch(model, optimizer, train_dataloader,
                                                       val_dataloader, device)
        savedir = instance_dir / 'checkpoints'
        if not savedir.exists(): savedir.mkdir()
        torch.save(model.state_dict(), savedir / f'e{i}.model')
        
        if val_data[1] > best_val_auc_roc:
            best_val_auc_roc = val_data[1]
            patience_counter = 0
            best_state_dict = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter == patience:
                print("Early stopping!")
                model.load_state_dict(best_state_dict)
                return None

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


def _train_classifier_epoch(model: torch.nn.Module,
                            optimizer: Optimizer,
                            train_dataloader: DataLoader,
                            val_dataloader: DataLoader,
                            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                            epoch=None, iteration=None, writer=None, lr_scheduler=None
                            ) -> Tuple[Tuple[List[Any], float],
                                       Tuple[List[Any], float]]:

    t_losses, t_true, t_pred = [], [], []
    v_losses, v_true, v_pred = [], [], []
    
    model.train()
    running_losses = []
    
    for x, y in train_dataloader:
        
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
        
        if (iteration[0] + 1) % 10 == 0:
             t_loss = sum(running_losses) / len(running_losses)
             running_losses = []
             print(f'[{epoch}, {iteration[0]}] -- Loss: {t_loss}')
             writer.add_scalar('HighFreqLoss', t_loss, iteration[0])
             lr_scheduler.step(t_loss)
        
        iteration[0] += 1

    with torch.no_grad():
        model.eval()
        num = 1
        for val_x, val_y in tqdm(val_dataloader):
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            
            val_logits = model(val_x)
            val_loss = F.cross_entropy(val_logits, val_y)
            v_losses.append(val_loss.item())

            val_preds = F.softmax(val_logits, -1)[:, 1]
            for i in range(val_x.shape[0]):
                
                image = val_x[i, :, :, :].cpu()
                image = denormalize(image.numpy())
                img_name = f'tmp_plots/{num}_{int((val_preds[i].item() > 0.5) * 1)}_{int(val_y[i])}.png'
                image = image.transpose((1, 2, 0))
                cv2.imwrite(str(img_name), image[:, :, ::-1])
                num += 1

            v_true.append(val_y.detach().cpu().numpy())
            v_pred.append(val_preds.detach().cpu().numpy())
    
    v_true = np.concatenate(v_true)
    v_pred = np.concatenate(v_pred)
    
    train_auc = roc_auc_score(np.concatenate(t_true), np.concatenate(t_pred))
    val_auc = roc_auc_score(v_true,v_pred)
    fpr, tpr, thresholds = roc_curve(v_true, v_pred)
    youden_j = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_j)]
    print("Best Threshold:", best_threshold)
    
    gt = v_true
    pd = (v_pred > 0.5) * 1
    pd2 = (v_pred > best_threshold) * 1
    
    print(classification_report(gt, pd, target_names=['empty', 'solar']))
    print(classification_report(gt, pd2, target_names=['empty', 'solar']))
    
    print(f'Train loss: {np.mean(t_losses)}, Train AUC ROC: {train_auc}, '
          f'Val loss: {np.mean(v_losses)}, Val AUC ROC: {val_auc}')
    
    writer.add_scalar('Loss/Train', np.mean(t_losses), epoch)
    writer.add_scalar('Loss/Validation', np.mean(v_losses), epoch)
    writer.add_scalar('AUC/Train', train_auc, epoch)
    writer.add_scalar('AUC/Validation', val_auc, epoch)
    
    print(f'Evaluate on london validation set with thresh 0.50 ...')
    prec, rec, f1, acc = evaluate_on_london(model, device, 0.5)
    
    print(f'Evaluate on london validation set with thresh {best_threshold:.2f} ...')
    prec, rec, f1, acc = evaluate_on_london(model, device, best_threshold)

    writer.add_scalar('LondonTestSet/Precision', prec, epoch)
    writer.add_scalar('LondonTestSet/Recall', rec, epoch)
    writer.add_scalar('LondonTestSet/F1', f1, epoch)
    writer.add_scalar('LondonTestSet/Accuracy', acc, epoch)

    return (t_losses, train_auc), (v_losses, val_auc)

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

def evaluate_on_london(model, device, thresh):

    classifier = model

    # When we have a group of 224x224 images            
    save_path = Path('data/test_manually_labeled_output')
    if not save_path.exists(): save_path.mkdir()
    ds = TestDataset('data/test_manually_labeled_input', MEAN=[0.5, 0.5, 0.5], STD=[0.5, 0.5, 0.5])
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    classifier.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for i, x in enumerate(dl):
            names = x[0]
            images = x[1].to(device)
            label = x[2].to(device)
            logit = classifier(images)
            pred = F.softmax(logit, -1)[:, 1]
            # print(names, "pred:", pred, "|", "label:", label)
            original_image = denormalize(images.cpu().numpy())
            pos_neg = f'{"P" if pred > thresh else "N"}'
            suffix = f'{pos_neg+"P" if label > 0.5 else pos_neg+"N"}'
            if pred > 0.5 and label == 1: tp += 1
            elif pred > 0.5 and label == 0: fp += 1
            elif pred < 0.5 and label == 1: fn += 1
            else: tn += 1
            img_name = save_path / f'{names[0].split("/")[-1].split(".")[0]}-{suffix}.png'
            original_image = original_image[0, :, :, :].transpose((1, 2, 0))
            # cv2.imwrite(str(img_name), original_image[:, :, ::-1])

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
        
    return precision, recall, f1, accuracy