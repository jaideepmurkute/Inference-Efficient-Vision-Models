from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image


class NeuDetDataset(Dataset):
    def __init__(self, img_paths, cls_ids, split_type='train', transform=None):
        self.img_paths = img_paths
        self.cls_ids = cls_ids
        self.split_type = split_type

        # Default transform if none provided
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        cls_id = self.cls_ids[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, cls_id
