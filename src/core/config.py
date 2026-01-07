from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

# Определение корневой директории проекта
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

@dataclass
class ModelConfig:
    """Конфигурация модели Mistral 7B."""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    load_in_4bit: bool = True
    compute_dtype: str = "float16"
    use_double_quant: bool = True
    device_map: str = "auto"

@dataclass
class LoRAConfig:
    """Конфигурация параметров LoRA."""
    r: int = 64  # Ранг низкоранговых матриц
    lora_alpha: int = 128  # Масштабирующий коэффициент
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

@dataclass
class TrainingConfig:
    """Параметры обучения LoRA-адаптеров."""
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    bf16: bool = True
    max_grad_norm: float = 0.3
    max_seq_length: int = 1024
    optim: str = "paged_adamw_8bit"
    report_to: str = "none"  # "wandb" для мониторинга

@dataclass
class GenerationConfig:
    """Параметры генерации текста."""
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.15
    num_beams: int = 1

@dataclass
class PathConfig:
    """Конфигурация путей к данным."""
    # Директории данных
    data_dir: Path = DATA_DIR
    train_data_dir: Path = DATA_DIR / "train_data"
    datasets_dir: Path = DATA_DIR / "datasets"
    adapters_dir: Path = DATA_DIR / "adapters"
    
    # Директория вывода
    output_dir: Path = OUTPUT_DIR / "generated_tickets"
    
    def __post_init__(self):
        """Создает необходимые директории."""
        for directory in [
            self.data_dir,
            self.train_data_dir,
            self.datasets_dir,
            self.adapters_dir,
            self.output_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)

@dataclass
class DomainConfig:
    """Конфигурация предметных областей."""
    domains: Dict[str, Dict] = None
    num_examples_per_domain: int = 500
    
    def __post_init__(self):
        if self.domains is None:
            adapters_dir = DATA_DIR / "adapters"
            self.domains = {
                "crypto": {
                    "file_patterns": ["cryptography", "crypto"],
                    "display_name": "криптографии",
                    "adapter_path": adapters_dir / "crypto"
                },
                "networks": {
                    "file_patterns": ["networks", "network", "сети"],
                    "display_name": "компьютерным сетям",
                    "adapter_path": adapters_dir / "networks"
                },
                "algorithms": {
                    "file_patterns": ["algorithms", "алгоритмы", "structures"],
                    "display_name": "алгоритмам и структурам данных",
                    "adapter_path": adapters_dir / "algorithms"
                }
            }

# Глобальные экземпляры конфигурации
MODEL_CONFIG = ModelConfig()
LORA_CONFIG = LoRAConfig()
TRAINING_CONFIG = TrainingConfig()
GENERATION_CONFIG = GenerationConfig()
PATH_CONFIG = PathConfig()
DOMAIN_CONFIG = DomainConfig()
