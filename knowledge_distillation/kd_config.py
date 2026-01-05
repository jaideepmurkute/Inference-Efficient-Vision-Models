import os
import torch

class KDConfig:
    def __init__(self, **kwargs):
        # Default configuration
        self.choice = 2 # 1: Train, 2: Test
        self.num_folds = 2
        
        self.experiment_name = "default_kd_exp"
        self.teacher_model = "vit_base_patch16_224"
        self.student_model = "vit_tiny_patch16_224" # Example student
        self.output_root = "output"
        self.batch_size = 4
        self.learning_rate = 1e-4
        self.epochs = 2
        self.num_classes = 6
        self.image_size = (224, 224)
        self.test_ckpt_type = "best" # 'best' or 'last'
        self.teacher_checkpoint = None

        self.data_dir = os.path.join("..", "data", "NEU-DET")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42
        self.num_workers = 2
        self.alpha = 0.5 # Balance between CE and KL
        self.temperature = 4.0 # Temperature for Softmax
        
        self.cls_name_id_map = {'crazing': 0, 'inclusion': 1, 'patches': 2, 'pitted_surface': 3, 
                    'rolled-in_scale': 4, 'scratches': 5}
        
        # Override defaults
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            
        # Create output directory
        self.output_dir = os.path.join(self.output_root, self.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def __repr__(self):
        return str(self.__dict__)
