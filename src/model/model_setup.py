"""
Модуль для загрузки и конфигурации Mistral 7B с LoRA.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from src.core.config import MODEL_CONFIG, LORA_CONFIG
from src.utils import logger

class ModelManager:
    """Управление загрузкой и настройкой модели."""
    
    @staticmethod
    def create_quantization_config():
        """Создает конфигурацию 4-битной квантизации."""
        return BitsAndBytesConfig(
            load_in_4bit=MODEL_CONFIG.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, MODEL_CONFIG.compute_dtype),
            bnb_4bit_use_double_quant=MODEL_CONFIG.use_double_quant,
        )
    
    @staticmethod
    def load_base_model():
        """
        Загружает базовую модель Mistral 7B.
        
        Returns:
            Tuple[model, tokenizer]: Модель и токенизатор
        """
        logger.info(f"Загрузка модели {MODEL_CONFIG.model_name}...")
        
        bnb_config = ModelManager.create_quantization_config()
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG.model_name,
            quantization_config=bnb_config,
            device_map=MODEL_CONFIG.device_map,
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG.model_name,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        logger.info("✓ Базовая модель загружена")
        return model, tokenizer
    
    @staticmethod
    def setup_lora_model():
        """
        Загружает модель и применяет LoRA.
        
        Returns:
            Tuple[model, tokenizer, lora_config]
        """
        model, tokenizer = ModelManager.load_base_model()
        
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=LORA_CONFIG.r,
            lora_alpha=LORA_CONFIG.lora_alpha,
            target_modules=LORA_CONFIG.target_modules,
            lora_dropout=LORA_CONFIG.lora_dropout,
            bias=LORA_CONFIG.bias,
            task_type=LORA_CONFIG.task_type,
        )
        
        model = get_peft_model(model, lora_config)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        logger.info(
            f"✓ LoRA: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%) параметров"
        )
        
        return model, tokenizer, lora_config
