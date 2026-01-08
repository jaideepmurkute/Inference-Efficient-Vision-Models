import os
import torch

class PruningConfig:
    def __init__(self, **kwargs):
        self.choice = 2 # 1: Prune+Finetune

        # Default configuration
        self.experiment_name = "kaggle_pruning_exp_2" 
        
        # Model Source
        self.source_exp_name = "kaggle_kd_exp_2"
        # self.source_exp_name = "kaggle_exp_2"
        
        # Paths to Source Models 
        # (Relative to pruning/ directory, assuming standard structure)
        self.student_exp_path = os.path.join("..", "knowledge_distillation", "output", self.source_exp_name)
        # self.student_exp_path = os.path.join("..", "teacher_training", "output", self.source_exp_name)
        
        # Model Architecture
        self.model_name = "resnet18" 
        # self.model_name = "resnet50" 
        
        self.num_classes = 6
        self.image_size = (224, 224)
        
        self.num_folds = 5  
        self.fold_id = 0 
        
        # Pruning Hyperparameters
        self.pruning_ratio = 0.05 
        self.pruning_type = "structured" 
        self.pruning_method = "l2" # 'l1', 'random', 'l2', 'group_norm', 'taylor'
        self.global_pruning = False 
        
        # ResNet18 (64 channels) needs round_to=8 or similar to allow pruning.
        # 20% of 64 is 12.8. If round_to=64, nothing happens.
        self.round_to = 1 
        
        # Fine-tuning Hyperparameters
        self.finetune_epochs = 0 
        self.learning_rate = 1e-5 
        self.batch_size = 64
        self.output_root = "output"
        
        # Data
        self.data_dir = os.path.join("..", "data", "NEU-DET")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42
        self.num_workers = 2
        
        # Class Map
        self.cls_name_id_map = {'crazing': 0, 'inclusion': 1, 'patches': 2, 'pitted_surface': 3, 
                    'rolled-in_scale': 4, 'scratches': 5}
        
        # Override defaults
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            
        # Create output directory
        self.output_dir = os.path.join(self.output_root, self.experiment_name)
        self.log_dir = self.output_dir 
        os.makedirs(self.output_dir, exist_ok=True)
        
    def __repr__(self):
        return str(self.__dict__)
