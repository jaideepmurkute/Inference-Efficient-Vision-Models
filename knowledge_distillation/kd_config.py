import os
import torch


class KDConfig:
    def __init__(self, **kwargs):
        self.choice = 2  # 1: Train, 2: Test

        self.experiment_name = "test"  # "kaggle_kd_exp_2"  # "test"
        self.teacher_exp_name = "kaggle_exp_2"  # "exp_1"
        self.DEBUG_MODE = True  # True for debugging - uses less data/epochs/batches

        self.teacher_model = "resnet50"  # "vit_base_patch16_224"
        self.student_model = "resnet18"  # "vit_tiny_patch16_224"
        self.use_timm = False

        self.alpha = 0.5  # Balance between classification-CE and teacher-KL losses
        self.temperature = 4.0  # Temperature for Softmax sharpening of teacher's logits

        self.num_folds = 5
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.epochs = 2
        self.num_classes = 6
        self.image_size = (224, 224)
        self.test_ckpt_type = "best"  # 'best' or 'last'
        self.teacher_checkpoint = None

        self.output_root = "output"
        self.data_dir = os.path.join("..", "data", "NEU-DET")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42
        self.num_workers = 2

        self.cls_name_id_map = {
            "crazing": 0,
            "inclusion": 1,
            "patches": 2,
            "pitted_surface": 3,
            "rolled-in_scale": 4,
            "scratches": 5,
        }

        # Override defaults if passed as kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.output_dir = os.path.join(self.output_root, self.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)

        if self.DEBUG_MODE:
            self.epochs = 2
            self.batch_size = 2
            self.num_folds = 3

    def __repr__(self):
        return str(self.__dict__)
