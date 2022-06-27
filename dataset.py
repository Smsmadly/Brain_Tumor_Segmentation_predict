''' dataset.py - Loads datasets and provides easy access through a dedicated class '''

import os
import random
import nibabel as nib
import numpy as np

class BraTS2018():
    
    def __init__(self, data_dir, batch_size=4):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def norm(self, image):
        ''' Normalize Image '''
        return (image - np.min(image)) / (np.max(image) - np.min(image))
        
    def process(self, sample, idx:int):
        ''' Resize sample to the given size '''
        sample = np.transpose(sample, (2, 0, 1))
        return sample[idx]
        
    def load(self, glioma='HGG', slice_idx='random', shuffle=True):
        assert glioma in ["HGG", "LGG"]
        batch_size = self.batch_size
        # Init
        batch = np.zeros((batch_size, 8, 240, 240))
        targets = np.zeros((batch_size, 1, 240, 240))
        base_dir = self.data_dir + os.path.sep + glioma
        samples = os.listdir(base_dir)
        if shuffle: random.shuffle(samples)
        # Yield samples when batch is full
        i = 0
        for sample in samples:
            sample_dir = base_dir + os.path.sep + sample
            si = slice_idx
            if slice_idx == 'random': 
                si = random.randint(90, 110)
            # Read samples
            t1w = nib.load(sample_dir + os.path.sep + f"{sample}_t1.nii").get_fdata().astype('float32')
            t1wpc = nib.load(sample_dir + os.path.sep + f"{sample}_t1ce.nii").get_fdata().astype('float32')
            t2w = nib.load(sample_dir + os.path.sep + f"{sample}_t2.nii").get_fdata().astype('float32')
            flair = nib.load(sample_dir + os.path.sep + f"{sample}_flair.nii").get_fdata().astype('float32')
            seg = nib.load(sample_dir + os.path.sep + f"{sample}_seg.nii").get_fdata().astype('float32')
            # Add to batch
            # Add to batch
            batch[i%batch_size, 0] = self.norm(self.process(t1w, si))
            batch[i%batch_size, 1] = self.norm(self.process(t1w, si) ** 2)
            batch[i%batch_size, 2] = self.norm(self.process(t1wpc, si))
            batch[i%batch_size, 3] = self.norm(self.process(t1wpc, si) ** 2)
            batch[i%batch_size, 4] = self.norm(self.process(t2w, si))
            batch[i%batch_size, 5] = self.norm(self.process(t2w, si) ** 2)
            batch[i%batch_size, 6] = self.norm(self.process(flair, si))
            batch[i%batch_size, 7] = self.norm(self.process(flair, si) ** 2)
            targets[i%batch_size, 0] = self.process(seg, si) 
            # Yield when batch is full
            i += 1
            if i > 0 and (i%batch_size) == 0:
                yield batch, targets