import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class BinaryMalwareDataset(Dataset):
    """
    Wrapper for ImageFolder to group all malware into one class.
    Class 0: Benign
    Class 1: Malware (all others)
    """
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.benign_idx = original_dataset.class_to_idx.get('Benign')
        if self.benign_idx is None:
            self.benign_idx = 0

    def __getitem__(self, index):
        img, label = self.original_dataset[index]
        new_label = 0 if label == self.benign_idx else 1
        return img, new_label

    def __len__(self):
        return len(self.original_dataset)

class LabelRemapDataset(Dataset):
    """
    Wraps a Subset or Dataset and remaps labels to be continuous 0 to N-1.
    Used for malware_only mode where some classes are skipped.
    """
    def __init__(self, original_dataset, class_names, full_ds_classes):
        self.original_dataset = original_dataset
        # Create a mapping from old index to new index
        self.mapping = {}
        for new_idx, class_name in enumerate(class_names):
            old_idx = full_ds_classes.index(class_name)
            self.mapping[old_idx] = new_idx

    def __getitem__(self, index):
        img, old_label = self.original_dataset[index]
        return img, self.mapping[old_label]

    def __len__(self):
        return len(self.original_dataset)

def get_dataloaders(root_dir, batch_size=32, img_size=224, mode='family', num_workers=4):
    """
    Creates DataLoaders for train, val, and test splits.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]),
    }

    image_datasets = {}
    final_class_names = None

    for x in ['train', 'val', 'test']:
        dir_path = os.path.join(root_dir, x)
        if not os.path.exists(dir_path):
            continue
            
        full_ds = datasets.ImageFolder(dir_path, data_transforms[x])
        
        if mode == 'binary':
            image_datasets[x] = BinaryMalwareDataset(full_ds)
            final_class_names = ['Benign', 'Malware']
        elif mode == 'malware_only':
            # 1. Filter out Benign indices
            benign_idx = full_ds.class_to_idx.get('Benign')
            indices = [i for i, (_, label) in enumerate(full_ds.samples) if label != benign_idx]
            subset = torch.utils.data.Subset(full_ds, indices)
            
            # 2. Identify the malware-only classes
            malware_classes = [c for c in full_ds.classes if c != 'Benign']
            if final_class_names is None:
                final_class_names = malware_classes
            
            # 3. Wrap to remap labels (e.g., 1-9 becomes 0-8)
            image_datasets[x] = LabelRemapDataset(subset, final_class_names, full_ds.classes)
            
        else: # mode == 'family'
            image_datasets[x] = full_ds
            if final_class_names is None:
                final_class_names = full_ds.classes

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=num_workers)
        for x in image_datasets.keys()
    }
    
    return dataloaders, final_class_names
