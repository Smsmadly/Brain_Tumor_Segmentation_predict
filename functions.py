# Loss Functions
import torch , os
import torch.nn as nn
torch.set_default_dtype(torch.float32)
torch.manual_seed(42)

# Dice
def DiceScore(predictions, targets, smooth=0.001):
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    intersection = (predictions * targets).sum()
    dice = (2 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    return dice

def DiceLoss(predictions, targets, smooth=1):
    dice = DiceScore(predictions, targets)
    return -1 * dice

# Focal
def FocalLoss(predictions, targets, alpha=0.7, gamma=4/3, smooth=0.001):
    BCE = nn.functional.binary_cross_entropy(predictions, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE
    return focal_loss
    
# FocalTversky
def FocalTverskyLoss(predictions, targets, alpha=0.7, beta=0.3, gamma=4/3, smooth=0.001):
    TP = (predictions * targets).sum()    
    FP = ((1 - targets) * predictions).sum()
    FN = (targets * (1 - predictions)).sum()
    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)  
    FocalTversky = (1 - Tversky) ** gamma
    return FocalTversky

# IntersectionOverUnion
def IOU(preds, targets, smooth=0.001):
    preds = preds.view(-1)
    targets = targets.view(-1)
    # Intersection is equivalent to True Positive count
    # Union is the mutually inclusive area of all labels & predictions 
    intersection = (preds & targets).float().sum()
    union = (preds | targets).float().sum()
    # Compute Score
    IoU = (intersection + smooth) / (union + smooth)
    return IoU

from pathlib import Path

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path