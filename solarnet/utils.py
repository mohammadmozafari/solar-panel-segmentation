import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve

def report_validation_results(t_losses, t_true, t_pred, v_losses, v_true, v_pred,
                              writer, epoch):
    
    train_auc = roc_auc_score(np.concatenate(t_true), np.concatenate(t_pred))
    val_auc = roc_auc_score(v_true, v_pred)
    fpr, tpr, thresholds = roc_curve(v_true, v_pred)
    youden_j = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_j)]
    print("Best Threshold:", best_threshold)
    
    gt = v_true
    pd1 = (v_pred > 0.5) * 1
    pd2 = (v_pred > best_threshold) * 1
    
    print(classification_report(gt, pd1, target_names=['empty', 'solar']))
    print(classification_report(gt, pd2, target_names=['empty', 'solar']))
    
    print(f'Train loss: {np.mean(t_losses)}, Train AUC ROC: {train_auc}, '
          f'Val loss: {np.mean(v_losses)}, Val AUC ROC: {val_auc}')
    
    writer.add_scalar('Loss/Train', np.mean(t_losses), epoch)
    writer.add_scalar('Loss/Validation', np.mean(v_losses), epoch)
    writer.add_scalar('AUC/Train', train_auc, epoch)
    writer.add_scalar('AUC/Validation', val_auc, epoch)
    
    return best_threshold