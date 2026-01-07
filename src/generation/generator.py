"""
Модуль для генерации экзаменационных билетов.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.core.config import MODEL_CONFIG, GENERATION_CONFIG, PATH_CONFIG, DOMAIN_CONFIG
from src.utils import logger

class ExamTicketGenerator:
    """Генератор экзаменационных билетов с LoRA-адаптерами."""
    
    def __init__(self):
        self.model_name = MODEL_CONFIG.model_name
        self.gen_config = GENERATION_CONFIG
        self.path_config = PATH_CONFIG
        self.domain_config = DOMAIN_CONFIG
        
        self.current_domain = None
        self.model = None
        self.tokenizer = None
    
    def load_adapter(self, domain_key: str):
        """
        Загружает LoRA-адаптер для домена.
        
        Args:
            domain_key: Ключ домена
        """
        if domain_key not in self.domain_config.domains:
            available = list(self.domain_config.domains.keys())
            raise ValueError(
                f"Домен '{domain_key}' не поддерживается. Доступные: {available}"
            )
        
        if self.current_domain == domain_key and self.model is not None:
            logger.info(f"Адаптер '{domain_key}' уже загружен")
            return
        
        logger.info(f"Загрузка адаптера: {domain_key}")
        
        # Освобождение памяти
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        
        adapter_path = self.domain_config.domains[domain_key]['adapter_path']
        
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Адаптер не найден: {adapter_path}\n"
                "Необходимо сначала обучить адаптеры."
            )
        
        # Квантизация
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # Базовая модель
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Адаптер
        self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
        self.model.eval()
        
        # Токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.current_domain = domain_key
        logger.info(f"✓ Адаптер '{domain_key}' загружен")
    
    def generate_ticket(self, domain_key: str, num_questions: int = 3,
                       ticket_number: int = 1, additional_info: str = "") -> str:
        """
        Генерирует экзаменационный билет.
        
        Args:
            domain_key: Ключ домена
            num_questions: Количество вопросов
            ticket_number: Номер билета
            additional_info: Дополнительная информация
        
        Returns:
            Сгенерированный билет
        """
        self.load_adapter(domain_key)
        
        domain_name = self.domain_config.domains[domain_key]['display_name']
        
        instruction = (
            f"Сгенерируй экзаменационный билет №{ticket_number} "
            f"по предмету '{domain_name}' с {num_questions} вопросами открытого типа"
        )
        
        if additional_info:
            prompt = (
                f"<s>[INST] {instruction}\n\n"
                f"Дополнительная информация: {additional_info} [/INST]"
            )
        else:
            prompt = f"<s>[INST] {instruction} [/INST]"
        
        # Токенизация
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Генерация
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_config.max_new_tokens,
                temperature=self.gen_config.temperature,
                top_p=self.gen_config.top_p,
                do_sample=self.gen_config.do_sample,
                repetition_penalty=self.gen_config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Декодирование
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        ticket = generated.split("[/INST]")[-1].strip()
        
        return ticket
    
    def generate_multiple_tickets(self, domain_key: str, num_questions: int = 3,
                                  num_tickets: int = 5, additional_info: str = "") -> list:
        """Генерирует несколько билетов."""
        tickets = []
        
        logger.info(f"Генерация {num_tickets} билетов ({domain_key})...")
        
        for i in range(1, num_tickets + 1):
            logger.info(f"Билет {i}/{num_tickets}...")
            ticket = self.generate_ticket(domain_key, num_questions, i, additional_info)
            tickets.append(ticket)
        
        logger.info(f"✓ Сгенерировано {len(tickets)} билетов")
        return tickets
    
    def save_tickets(self, tickets: list, domain_key: str, filename: str = None) -> Path:
        """Сохраняет билеты в файл."""
        output_dir = self.path_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"{domain_key}_tickets.txt"
        
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, ticket in enumerate(tickets, 1):
                f.write(f"{'='*60}\n")
                f.write(f"БИЛЕТ {i}\n")
                f.write(f"{'='*60}\n")
                f.write(ticket + "\n\n")
        
        logger.info(f"✓ Билеты сохранены: {filepath}")
        return filepath
