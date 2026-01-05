import os
import torch

class TeacherConfig:
    def __init__(self, **kwargs):
        # Default configuration
        self.choice = 2 # 1: Train, 2: Test
        self.experiment_name = "exp_1"
        
        self.model_name = "vit_base_patch16_224"
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.epochs = 3
        self.num_classes = 6
        self.image_size = (224, 224)
        self.test_ckpt_type = "best" # 'best' or 'last'
        self.num_folds = 2
        
        self.data_dir = os.path.join("..", "data", "NEU-DET")
        self.output_root = "output"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42
        self.num_workers = 2
        
        self.cls_name_id_map = {'crazing': 0, 'inclusion': 1, 'patches': 2, 'pitted_surface': 3, 
                    'rolled-in_scale': 4, 'scratches': 5}
        
        # Override defaults with provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            
        # Create output directory
        self.output_dir = os.path.join(self.output_root, self.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def __repr__(self):
        return str(self.__dict__)
