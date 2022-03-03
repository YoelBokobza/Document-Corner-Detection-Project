from torch.utils.data import Dataset      
from torch.utils.data import DataLoader
import torch  
import torch.optim as optim            
import torch.nn as nn                        
import torch.nn.functional as F  
from utils.classic_image_processing_algorithms import e2e_algorithm
from utils.utils import gen_mask
from utils.dice_score import dice_coeff
import numpy as np

device = torch.device('cpu')

def evaluate_e2e_algorithm(dataset, batch_size, net):
    loader_args = dict(batch_size=batch_size, pin_memory=True)
    loader = DataLoader(dataset, shuffle=True, drop_last=True, **loader_args)
    cnt = 0
    dice_score = []
    for batch in loader:
        images = batch['image']
        true_masks = batch['mask']
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        with torch.no_grad():
            masks_pred = net(images)
        predSegment = masks_pred[:, 1, ...]
        predSegment[predSegment < 0.5] = 0
        predSegment[predSegment >= 0.5] = 1
        temp_dice_score = []
        for idx in range(batch_size):
            curr_cleaned_mask = predSegment[idx].detach().cpu().numpy()
            masks_pred = e2e_algorithm(curr_cleaned_mask)[0]
            mask_predicated = gen_mask(masks_pred, sizeX=true_masks[idx].shape[0], sizeY=true_masks[idx].shape[1], sort=True)
            score = dice_coeff(torch.tensor(mask_predicated, dtype=torch.float).to(device), true_masks[idx].type(torch.float))
            # print(score.item())
            temp_dice_score.append(score.item())
        # print(f'Batch-{cnt} - Dice Score = {np.mean(temp_dice_score)}')
        dice_score.extend(temp_dice_score)
        cnt += 1
    return np.array(dice_score)
