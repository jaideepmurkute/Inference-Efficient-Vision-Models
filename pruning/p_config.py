import os
import torch


class PruningConfig:
    def __init__(self, **kwargs):
        self.choice = 1  # 1: Prune+Finetune; 2: Test

        self.experiment_name = "test"  # "kaggle_pruning_exp_2"

        # Model Source
        self.source_exp_name = "kaggle_kd_exp_2"

        # Paths to Source Models - relative to pruning/ directory
        self.student_exp_path = os.path.join(
            "..", "knowledge_distillation", "output", self.source_exp_name
        )
        # self.student_exp_path = os.path.join("..", "teacher_training", "output", self.source_exp_name)

        self.DEBUG_MODE = False

        # Source model Architecture
        self.model_name = "resnet18"  # "resnet50"
        self.num_classes = 6
        self.image_size = (224, 224)
        self.num_folds = 5  # Same as source model
        # self.fold_id = 0

        # Pruning Hyperparameters
        self.pruning_ratio = 0.05
        self.pruning_type = "structured"
        self.pruning_method = "l2"  # 'l1', 'random', 'l2', 'group_norm', 'taylor'
        self.global_pruning = False
        self.round_to = 1  # Rounds up channels/layers to be pruned to the nearest multiple of round_to

        # Fine-tuning Hyperparameters
        self.finetune_epochs = 0  # 0: no fine-tuning
        self.learning_rate = 1e-5
        self.batch_size = 64
        self.output_root = "output"

        # Data
        self.data_dir = os.path.join("..", "data", "NEU-DET")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42
        self.num_workers = 2

        # Class Map
        self.cls_name_id_map = {
            "crazing": 0,
            "inclusion": 1,
            "patches": 2,
            "pitted_surface": 3,
            "rolled-in_scale": 4,
            "scratches": 5,
        }

        # Override defaults
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Create output directory
        self.output_dir = os.path.join(self.output_root, self.experiment_name)
        self.log_dir = self.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        if self.DEBUG_MODE:
            self.num_folds = 1
            self.fold_id = 0
            self.finetune_epochs = 1

    def __repr__(self):
        return str(self.__dict__)
