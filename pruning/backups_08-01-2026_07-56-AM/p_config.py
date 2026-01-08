import os
import torch

class PruningConfig:
    def __init__(self, **kwargs):
        self.choice = 1 # 1: Prune+Finetune

        # Default configuration
        self.experiment_name = "test" # "default_pruning_exp"
        
        # Model Source
        self.model_source = "student" # 'teacher', 'student', or 'custom'
        self.source_exp_name = "kaggle_kd_exp_2"
        self.custom_model_path = None # Only used if model_source is 'custom'

        # Paths to Source Models (Assumes standard project structure)
        self.teacher_exp_path = os.path.join("..", "teacher_training", "output", self.source_exp_name)
        self.student_exp_path = os.path.join("..", "knowledge_distillation", "output", self.source_exp_name)
        
        # Model Architecture 
        self.model_name = "resnet18" # "vit_tiny_patch16_224"
        self.use_timm = False
        self.num_classes = 6
        self.image_size = (224, 224)
        
        self.num_folds = 5  
        self.fold_id = 0 # not used
        
        
        # Pruning Hyperparameters
        self.pruning_ratio = 0.2 # Prune 20% of channels
        self.pruning_type = "structured" # CNNs love structured pruning!
        self.pruning_method = "l1" # 'l1', 'random', etc.
        self.global_pruning = False # Set to False to prevent layer collapse (e.g. 1 channel).
                                    # Ensures every layer retains (1-ratio)% of its capacity.
        self.round_to = 8 # Round channels to nearest multiple of 8. 
                          # Lower value needed for ResNet18 (64 channels) to allow pruning.
                          # Was 64 for ViT heads.
        
        # Fine-tuning Hyperparameters
        self.finetune_epochs = 2 # Quick recovery
        self.learning_rate = 1e-5 # Lower LR to prevent destroying weights (was 5e-5)
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
        self.log_dir = self.output_dir # Alias for main.py usage
        os.makedirs(self.output_dir, exist_ok=True)
        
    def __repr__(self):
        return str(self.__dict__)
