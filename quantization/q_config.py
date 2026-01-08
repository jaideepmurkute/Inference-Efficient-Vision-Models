import os
import torch

class QuantConfig:
    def __init__(self, **kwargs):
        # Default configuration
        self.experiment_name = "test_quant_exp"
        
        self.model_type = "pruned" # 'teacher', 'student', or 'pruned'
        self.student_model = "resnet18" # "vit_tiny_patch16_224"
        self.teacher_model = "resnet50" # "vit_base_patch16_224"
        
        # self.pruned_model_name = "resnet18" # Architecture of the pruned model
        self.pruned_model_name = "test" # Architecture of the pruned model
        
        self.use_timm = False
        self.num_classes = 6
        self.image_size = (224, 224)
        
        # Data
        self.data_dir = os.path.join("..", "data", "NEU-DET")
        self.fold_id = 0 # Fold to use for evaluation
        
        # Paths to experimental outputs
        self.teacher_exp_path = os.path.join("..", "teacher_training", "output", "kaggle_exp_2")
        self.student_exp_path = os.path.join("..", "knowledge_distillation", "output", "kaggle_kd_exp_2")
        self.pruning_exp_path = os.path.join("..", "pruning", "output", "test")
        
        self.output_root = "output"
        
        self.batch_size = 32
        # Number of batches to use for calibration (Static Quantization)
        self.num_calibration_batches = 10 
        
        self.device = "cpu" # Quantization is typically evaluated on CPU
        self.seed = 42
        self.num_workers = 2
        
        # Re-using the same Class Map
        self.cls_name_id_map = {'crazing': 0, 'inclusion': 1, 'patches': 2, 'pitted_surface': 3, 
                    'rolled-in_scale': 4, 'scratches': 5}
        
        self.num_folds = 5 # Needed for create_fold_split_idx
        self.choice = 2 # Reuse logic, effectively 'Test' mode

        # Override defaults
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            
        # Create output directory
        self.output_dir = os.path.join(self.output_root, self.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def __repr__(self):
        return str(self.__dict__)
