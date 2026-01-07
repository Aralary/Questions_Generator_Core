import random
from pathlib import Path
from typing import List, Dict

from src.core.config import PATH_CONFIG, DOMAIN_CONFIG
from src.utils import (
    load_questions_from_file,
    validate_questions,
    save_jsonl,
    logger,
    print_section
)

class DatasetGenerator:
    """Генератор обучающих датасетов из базы вопросов."""
    
    def __init__(self):
        self.path_config = PATH_CONFIG
        self.domain_config = DOMAIN_CONFIG
        self.question_counts = [2, 3, 5]
        
        self.additional_infos = [
            "Уровень: бакалавриат, 3 курс",
            "Уровень: бакалавриат, 4 курс",
            "Сложность: средняя",
            "Сложность: высокая",
            "Уровень: магистратура",
            "", "", ""
        ]
    
    def generate_ticket(self, questions: List[str], num_questions: int,
                       ticket_number: int, domain_name: str) -> str:
        """
        Генерирует экзаменационный билет.
        
        Args:
            questions: Список доступных вопросов
            num_questions: Количество вопросов в билете
            ticket_number: Номер билета
            domain_name: Название предметной области
        
        Returns:
            Форматированный билет
        """
        if len(questions) < num_questions:
            raise ValueError(
                f"Недостаточно вопросов: требуется {num_questions}, "
                f"доступно {len(questions)}"
            )
        
        selected = random.sample(questions, num_questions)
        
        ticket = f"Экзаменационный билет №{ticket_number} по {domain_name}\n\n"
        for i, question in enumerate(selected, 1):
            ticket += f"{i}. {question}\n\n"
        
        return ticket.strip()
    
    def create_training_examples(self, questions: List[str],
                                 domain_name: str,
                                 num_examples: int) -> List[Dict]:
        """
        Создает обучающие примеры из базы вопросов.
        
        Args:
            questions: Список вопросов
            domain_name: Название домена
            num_examples: Количество примеров
        
        Returns:
            Список обучающих примеров
        """
        examples = []
        
        templates = [
            "Сгенерируй экзаменационный билет по {} с {} вопросами открытого типа",
            "Создай экзаменационный билет по {} с {} вопросами",
            "Составь билет по предмету '{}' на {} вопроса",
            "Подготовь экзаменационный билет по {} ({} вопроса)"
        ]
        
        for ticket_num in range(1, num_examples + 1):
            num_questions = random.choice(self.question_counts)
            
            ticket = self.generate_ticket(
                questions, num_questions, ticket_num, domain_name
            )
            
            template = random.choice(templates)
            instruction = template.format(domain_name, num_questions)
            additional_info = random.choice(self.additional_infos)
            
            examples.append({
                "instruction": instruction,
                "input": additional_info,
                "output": ticket
            })
        
        return examples
    
    def prepare_all_datasets(self):
        """Подготавливает датасеты для всех доменов."""
        print_section("ПОДГОТОВКА ОБУЧАЮЩИХ ДАТАСЕТОВ")
        
        if not self.path_config.train_data_dir.exists():
            raise FileNotFoundError(
                f"Папка {self.path_config.train_data_dir} не найдена!"
            )
        
        for domain_key, domain_info in self.domain_config.domains.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Обработка домена: {domain_key}")
            logger.info(f"{'='*60}")
            
            # Поиск файла
            found_file = self._find_domain_file(domain_key, domain_info)
            
            if not found_file:
                logger.warning(
                    f"Файл для '{domain_key}' не найден. "
                    f"Ожидаемые имена: {domain_info['file_patterns']}"
                )
                continue
            
            # Загрузка и валидация
            questions = load_questions_from_file(found_file)
            questions = validate_questions(questions)
            
            if len(questions) < 10:
                logger.warning(
                    f"Мало вопросов для {domain_key}: {len(questions)}. "
                    "Рекомендуется 50+"
                )
            
            # Генерация примеров
            logger.info(f"Генерация {self.domain_config.num_examples_per_domain} примеров...")
            examples = self.create_training_examples(
                questions,
                domain_info['display_name'],
                self.domain_config.num_examples_per_domain
            )
            
            # Сохранение
            output_file = self.path_config.datasets_dir / f"{domain_key}_train.jsonl"
            save_jsonl(examples, output_file)
            
            logger.info(
                f"✓ Датасет: {len(examples)} примеров "
                f"из {len(questions)} вопросов"
            )
        
        print_section("ПОДГОТОВКА ЗАВЕРШЕНА")
    
    def _find_domain_file(self, domain_key: str, domain_info: Dict) -> Path:
        """Ищет файл с вопросами для домена."""
        for pattern in domain_info['file_patterns']:
            for ext in ['.txt', '.json', '.jsonl']:
                filepath = self.path_config.train_data_dir / f"{pattern}{ext}"
                if filepath.exists():
                    return filepath
        return None
