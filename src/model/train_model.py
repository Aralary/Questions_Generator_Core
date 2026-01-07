"""
Модуль для обучения LoRA-адаптеров.
"""

import torch
from pathlib import Path
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

from src.model.model_setup import ModelManager
from src.core.config import TRAINING_CONFIG, PATH_CONFIG, DOMAIN_CONFIG
from src.utils import logger, print_section

class AdapterTrainer:
    """Класс для обучения LoRA-адаптеров."""
    
    def __init__(self):
        self.training_config = TRAINING_CONFIG
        self.path_config = PATH_CONFIG
        self.domain_config = DOMAIN_CONFIG
    
    @staticmethod
    def format_instruction(example):
        """Форматирует пример в формат Mistral Instruct."""
        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]
        
        if input_text:
            prompt = (
                f"<s>[INST] {instruction}\n\n"
                f"Дополнительная информация: {input_text} [/INST] "
                f"{output}</s>"
            )
        else:
            prompt = f"<s>[INST] {instruction} [/INST] {output}</s>"
        
        return {"text": prompt}
    

    def train_domain_adapter(self, domain_key: str, domain_info: dict):
        """Обучает адаптер для конкретного домена."""
        print_section(f"ОБУЧЕНИЕ АДАПТЕРА: {domain_key.upper()}")
        
        dataset_path = self.path_config.datasets_dir / f"{domain_key}_train.jsonl"
        output_dir = domain_info['adapter_path']
        
        if not dataset_path.exists():
            logger.error(f"Датасет не найден: {dataset_path}")
            return
        
        # Загрузка модели
        model, tokenizer, _ = ModelManager.setup_lora_model()
        
        # Загрузка датасета
        dataset = load_dataset('json', data_files=str(dataset_path), split='train')
        
        def formatting_func(example):
            """Форматирует ОДИН пример для обучения."""
            instruction = example['instruction']
            input_text = example['input']
            output = example['output']
            
            if input_text:
                text = f"<s>[INST] {instruction}\n\nДополнительная информация: {input_text} [/INST] {output}</s>"
            else:
                text = f"<s>[INST] {instruction} [/INST] {output}</s>"
            
            return text
        
        logger.info(f"Датасет: {len(dataset)} примеров")
        
        # Конфигурация обучения
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            optim=self.training_config.optim,
            learning_rate=self.training_config.learning_rate,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            warmup_ratio=self.training_config.warmup_ratio,
            logging_steps=self.training_config.logging_steps,
            save_strategy=self.training_config.save_strategy,
            save_total_limit=self.training_config.save_total_limit,
            bf16=self.training_config.bf16,
            max_grad_norm=self.training_config.max_grad_norm,
            report_to=self.training_config.report_to,
        )
        
        # SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            formatting_func=formatting_func,
            processing_class=tokenizer,
        )
        
        # Обучение
        logger.info("Запуск обучения...")
        trainer.train()
        
        # Сохранение
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        logger.info(f"✓ Адаптер сохранен: {output_dir}")
        
        # Очистка памяти
        del model, trainer
        torch.cuda.empty_cache()


    def train_all_adapters(self):
        """Обучает адаптеры для всех доменов."""
        print_section("ОБУЧЕНИЕ ВСЕХ АДАПТЕРОВ")
        
        for domain_key, domain_info in self.domain_config.domains.items():
            self.train_domain_adapter(domain_key, domain_info)
        
        print_section("ОБУЧЕНИЕ ЗАВЕРШЕНО")
