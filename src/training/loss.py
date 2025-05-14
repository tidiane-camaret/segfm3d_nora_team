import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, to_onehot_y=True, softmax=True, epsilon=1e-5):
        super().__init__()
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.epsilon = epsilon
        
    def forward(self, pred, target):
        if self.softmax:
            pred = torch.softmax(pred, dim=1)
            
        if self.to_onehot_y:
            n_classes = pred.shape[1]
            target = torch.nn.functional.one_hot(target, num_classes=n_classes)
            target = target.permute(0, 4, 1, 2, 3).float()  # For 3D data
            
        # Flatten predictions and targets
        pred = pred.flatten(2).float()
        target = target.flatten(2).float()
        
        # Calculate Dice score
        intersect = (pred * target).sum(-1)
        denominator = (pred * pred).sum(-1) + (target * target).sum(-1)
        dice_score = 2.0 * intersect / (denominator + self.epsilon)
        
        # Return loss
        return 1.0 - dice_score.mean()