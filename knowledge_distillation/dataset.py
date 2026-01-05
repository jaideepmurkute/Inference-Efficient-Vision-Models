import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class NeuDetDataset:
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        
        # Determine the root directory for the specific split
        split_dir = "train" if split == 'train' else "validation"
        self.images_dir = os.path.join(data_dir, split_dir, "images")
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.images_dir}")

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Use ImageFolder
        self.dataset = datasets.ImageFolder(root=self.images_dir, transform=self.transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def get_dataloader(data_dir, split, batch_size, num_workers=4, transform=None):
    dataset = NeuDetDataset(data_dir, split=split, transform=transform)
    shuffle = True if split == 'train' else False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader
