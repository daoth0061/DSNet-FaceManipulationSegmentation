import cv2
import numpy as np

from torch.utils import data
import torch

from torchvision import transforms
from torchvision.transforms import Normalize, ToTensor
import glob

import os
import inspect
# Create dataset class for face manipulation detection
class FaceManipulationDataset(data.Dataset):
    def __init__(self, fake_dir, real_dir, mask_dir, high_quality_images_path = None, split='train'):
        self.split = split
        self.transform = self.get_transform(split)
        self.transform_mask = self.get_transform_mask()
        
        # Get all image paths
        self.samples = []
        
        # Add real images (all have _0 suffix and no mask)
        real_images = sorted(glob.glob(os.path.join(real_dir, '*_0.*')))
        for img_path in real_images:
            img_name = os.path.basename(img_path)
            self.samples.append({
                'image': img_path,
                'mask': None,  # Real images have no mask (all zeros)
                'label': 0,    # Real = 0
                'is_real': True
            })
        
        # Load high_quality image list
        if high_quality_images_path:
            with open(high_quality_images_path, 'r') as f:
                self.high_quality_images = set(line.strip() for line in f.readlines())
        else:
            self.high_quality_images = set()
        
        # Add fake images (with corresponding masks)
        fake_images = sorted(glob.glob(os.path.join(fake_dir, '*.*')))
        for img_path in fake_images:
            img_name = os.path.basename(img_path)
            # Skip low-quality fake images
            if img_name not in self.high_quality_images:
                continue

            # Get name without extension
            name_without_ext = os.path.splitext(img_name)[0]

            # Find corresponding mask (check for both jpg and png)
            mask_jpg = os.path.join(mask_dir, f"{name_without_ext}.jpg")
            mask_png = os.path.join(mask_dir, f"{name_without_ext}.png")
            
            if os.path.exists(mask_jpg):
                mask_path = mask_jpg
            elif os.path.exists(mask_png):
                mask_path = mask_png
            else:
                # If no mask found, log the image path and continue
                # Get the current frame and line number for logging
                frame = inspect.currentframe()
                file_name = os.path.basename(inspect.getfile(frame))
                line_number = frame.f_lineno

                print(f"[WARN] Mask not found for: {img_path} (at {file_name}:{line_number})")
                continue
                
            self.samples.append({
                'image': img_path,
                'mask': mask_path,
                'label': 1,    # Fake = 1
                'is_real': False
            })
        
        print(f"[{split}] Loaded {len(self.samples)} samples: {sum(1 for s in self.samples if s['is_real'])} real, {sum(1 for s in self.samples if not s['is_real'])} fake")
    
    def get_transform(self, split):
        if split == 'train':
            return transforms.Compose([
                transforms.ToPILImage(), # Convert to PIL Image for augmentation
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
            ])
    def get_transform_mask(self):
        return transforms.Compose([
            transforms.ToPILImage(), # Convert to PIL Image for augmentation
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ToTensor(),
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        # Load image
        image = cv2.imread(sample['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        if (w, h) != (256, 256):
            image = cv2.resize(image, (256, 256))
        
        # Create or load mask
        if sample['is_real']:
            # For real images, create an all-zero mask
            mask = np.zeros((256, 256), dtype=np.float32)
        else:
            # For fake images, load the mask
            mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (256, 256))
            # Normalize mask to [0, 1]
            mask = mask.astype(np.float32) / 255.0
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform_mask(mask)
        
        # Convert mask to tensor if not already
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        
        # Add channel dimension to mask if needed
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'label': sample['label'],
            'path': sample['image']
        }