import importlib
from pathlib import Path
from typing import Dict, List, Optional, Self, Union

import yaml
from pydantic import BaseModel, Field

class Instanceable:
    class_path: str
    def get_instance(self):
        module_path, class_name = self.class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

class OptimizerConfig(BaseModel, Instanceable):
    class_path: str = Field(default='torch.optim.AdamW', description='Path to optimizer class e.g. \'torch.optim.AdamW\'')
    # optimizer_in_bwd: bool = False
    cpu_offload: bool = False
    params: Dict = Field(default_factory=dict)

class SchedulerConfig(BaseModel, Instanceable):
    class_path: str = Field('', description='Path to scheduler class')
    warmup_ratio: float = Field(0.1, ge=0.0, le=1.0)
    params: Dict = Field(default_factory=dict)

class KDLossConfig(BaseModel):
    gguf_path: str = Field('', description='Path to .gguf models')
    teacher_device: str = Field('cuda', description='teacher models device')
    kll_loss_ratio: float = Field(.5, gt=0., le=1., description='KD Loss Ratio')

class LossConfig(BaseModel):
    num_output_chunks: int = Field(1, gt=0)

class GradClipConfig(BaseModel):
    enabled: bool = False
    max_norm: float = Field(1.0, gt=0.0)

class TrainingConfig(BaseModel):
    compile: bool = False
    epochs: int = Field(1, gt=0)
    batch_size: int = Field(1, gt=0)
    gradient_accumulation_steps: int = Field(1, gt=0)
    offload_activations: bool = False
    offload_embeddings: bool = False
    device: str = Field('cuda', description='training device')
    checkpointing_layers: List[str] = Field(default_factory=list, description='List of layer names to apply gradient checkpointing')
    grad_clip: GradClipConfig = Field(default_factory=GradClipConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: Optional[SchedulerConfig] = None
    loss: LossConfig = Field(default_factory=LossConfig)
    kd: Optional[KDLossConfig] = None

class LoggingConfig(BaseModel):
    tensorboard_dir: str = Field('', description='path to save logs')
    log_frequency: int = Field(1, gt=0)
    save_frequency: int = Field(1, gt=0)

class DatasetConfig(BaseModel, Instanceable):
    class_path: str = Field('', description='Class path for dataset')
    max_steps: Optional[int] = Field(None, gt=0)
    params: Dict = Field(default_factory=dict)

class CollateFnConfig(BaseModel, Instanceable):
    class_path: str = Field('', description='Class path for CollateFN')
    params: Dict = Field(default_factory=dict)

class DataConfig(BaseModel):
    train_dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    valid_dataset: Optional[DatasetConfig] = None
    num_workers: int = Field(0, ge=0)
    pin_memory: bool = True
    collate_fn: Optional[CollateFnConfig] = None

class TrainerConfig(BaseModel):
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: Optional[LoggingConfig] = Field(default_factory=LoggingConfig)

    def save_config(self, path: Union[str, Path]) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.model_dump(), f)

    @classmethod
    def load_config(cls, path: Union[str, Path]) -> Self:
        with open(path, encoding='utf-8') as f:
            return cls(**yaml.safe_load(f))