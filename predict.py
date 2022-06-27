import matplotlib.pyplot as plt
# %matplotlib qt5
import time, cv2, os, os.path as osp, sys
import numpy as np
import random
random.seed(42)
import nibabel as nib
import torch
import torch.nn as nn
# from dataset import BraTS2018
from model import FCN
from functions import DiceScore, IOU, increment_path
from pathlib import Path

torch.set_default_dtype(torch.float32)
torch.manual_seed(42)

# Plot some samples and compare the results
batch_size = 1
n = (40 // batch_size) # Plot about 40 samples
cm = "inferno" # Color Map

in_channels = 8
out_channels = 1
model = FCN(in_channels, out_channels)
# model.cuda()
model.cpu()
model.eval()

PATH = 'basicUNET_final.torch'
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()
show_method = 'cv2'

def norm(image):
    ''' Normalize Image '''
    return (image - np.min(image)) / (np.max(image) - np.min(image))
    
def process(sample, idx:int):
    ''' Resize sample to the given size '''
    sample = np.transpose(sample, (2, 0, 1))
    return sample[idx]
    
def read(sample_dir,slice_idx=100):
    global sample
    
    sample = osp.basename(sample_dir)
    
    # si = random.randint(90, 110)
    si = slice_idx
    
    batch = np.zeros((batch_size, 8, 240, 240))
    # targets = np.zeros((batch_size, 1, 240, 240))
    
    # Read samples
    t1w   = nib.load(sample_dir + osp.sep + f"{sample}_t1.nii").get_fdata().astype('float32')
    t1wpc = nib.load(sample_dir + osp.sep + f"{sample}_t1ce.nii").get_fdata().astype('float32')
    t2w   = nib.load(sample_dir + osp.sep + f"{sample}_t2.nii").get_fdata().astype('float32')
    flair = nib.load(sample_dir + osp.sep + f"{sample}_flair.nii").get_fdata().astype('float32')
    # seg   = nib.load(sample_dir + osp.sep + f"{sample}_seg.nii").get_fdata().astype('float32')
    
    # Add to batch
    k=0
    batch[k%batch_size, 0] = norm(process(t1w, si))
    batch[k%batch_size, 1] = norm(process(t1w, si) ** 2)
    batch[k%batch_size, 2] = norm(process(t1wpc, si))
    batch[k%batch_size, 3] = norm(process(t1wpc, si) ** 2)
    batch[k%batch_size, 4] = norm(process(t2w, si))
    batch[k%batch_size, 5] = norm(process(t2w, si) ** 2)
    batch[k%batch_size, 6] = norm(process(flair, si))
    batch[k%batch_size, 7] = norm(process(flair, si) ** 2)
    # targets[k%batch_size, 0] = BraTS2018.process(seg, si) 
    
    # return batch, targets
    return batch

vid_path, vid_writer, vid_cap = None, None, None

# Directories
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
project = ROOT /'results'
name = 'exp'
# sample = 
save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run

os.makedirs(save_dir,exist_ok=True)
save_results = True
save_mode = 'video'
# save_mode = 'image'
        
sample_dir = r"brats2018_reduced\MICCAI_BraTS_2018_Data_Validation\Brats18_CBICA_AAM_1"
# sample_dir = r"brats2018_reduced\MICCAI_BraTS_2018_Data_Training\HGG\Brats18_2013_10_1"

n=155
# n=100
step = 3
# step = 30
for si in range(0,n,step):
    
    batch = read(sample_dir, slice_idx=si)
    batch = torch.tensor(batch)
    # targets = torch.tensor(targets)
    # targets_one_hot = targets > 0
    
    save_path = str(save_dir / sample)  # im.jpg
    
    # Copy to GPU
    if torch.cuda.is_available():
        batch = batch.float().cuda()
        # targets = targets_one_hot.float().cuda()
    else:
        batch = batch.float().cpu()
        # targets = targets_one_hot.float().cpu()
    
    # Forward Propagation
    with torch.no_grad():
        outputs = model(batch)
    # Plot Outputs
    print(f"sample : {sample}, slice_idx : {si}")
    for j in range(batch_size):
        flair = batch[j, 6].cpu()
        # targets = targets_one_hot[j, 0].cpu()
        predictions = outputs[j, 0].cpu()
        mapped_predictions = predictions > 0.75
        # iou_score = float(IOU(mapped_predictions, targets))
        # dice_score = float(DiceScore(mapped_predictions, targets))
        
        if show_method=='plot':
            # Plot these results
            fig, ax = plt.subplots(1, 4, figsize=(16, 16))
            ax[0].imshow(flair, cmap=cm) # Flair
            ax[0].set_title("Flair")
            # ax[1].imshow(targets, cmap=cm)
            ax[1].set_title("Targets")
            ax[2].imshow(predictions, cmap=cm)
            ax[2].set_title("Predictions")
            ax[3].imshow(mapped_predictions, cmap=cm)
            ax[3].set_title("Mapped Predictions")
            # plt.suptitle(f"IOU: {iou_score:.5f}  DICE: {dice_score:.5f}", fontsize=20, y=0.64)
            plt.show()
            
        elif show_method=='cv2':
            predictions_np = predictions.cpu().detach().numpy()
            
            flair_np = flair.cpu().detach().numpy()
            flair_np = cv2.cvtColor(flair_np,cv2.COLOR_GRAY2BGR)
            predictions_np = cv2.cvtColor(predictions_np,cv2.COLOR_GRAY2BGR)
            predictions_np[:,:,1] = predictions_np[:,:,0] = 0
            
            alpha_s = predictions_np[:, :, 2]
            alpha_l = 1.0 - alpha_s
            
            for c in range(0, 3):
                predictions_np[:, :, c] = (alpha_s * predictions_np[:, :, c] +
                                          alpha_l * flair_np[:, :, c])
            
            vis = np.concatenate((flair_np, predictions_np), axis=0)

            # resize image
            # vis = cv2.resize(vis, (640,360), interpolation = cv2.INTER_AREA)
            
            vis*=255
            vis = vis.astype(np.uint8)
            cv2.imshow('vis',vis)
            cv2.waitKey(1)
            
            # Save results (image with detections)

            if save_results:

                if save_mode == 'image':
                    cv2.imwrite(save_path+f'_si{si}.jpg', vis)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 5, vis.shape[1], vis.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(vis)
            
            
    # Clear Cache
    del batch
    # del targets
    del outputs
    torch.cuda.empty_cache()
    # break # Only process one batch for every 'i'

if save_mode == 'video':
    vid_writer.release()