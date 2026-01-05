import os
import torch

class QuantConfig:
    def __init__(self, **kwargs):
        # Default configuration
        self.experiment_name = "default_quant_exp"
        
        self.model_type = "student" # 'teacher' or 'student'
        self.student_model = "vit_tiny_patch16_224"
        self.teacher_model = "vit_base_patch16_224"
        self.num_classes = 6
        self.image_size = (224, 224)
        
        # Data
        self.data_dir = os.path.join("..", "data", "NEU-DET")
        self.fold_id = 0 # Fold to use for evaluation
        
        # Paths to experimental outputs
        self.teacher_exp_path = os.path.join("..", "teacher_training", "output", "exp_1")
        self.student_exp_path = os.path.join("..", "knowledge_distillation", "output", "default_kd_exp")
        
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
