import random
from pathlib import Path
from typing import List, Dict, Optional
from src.core.config import PATH_CONFIG, DOMAIN_CONFIG
from src.utils import (
    load_questions_from_file,
    validate_questions,
    save_jsonl,
    logger,
    print_section
)
# НОВЫЙ ИМПОРТ
from src.utils.lecture_parser import (
    load_lectures_for_domain,
    split_text_into_chunks,
    filter_relevant_chunks,
    extract_keywords
)


class DatasetGenerator:
    """Генератор обучающих датасетов из базы вопросов и лекций."""
    
    def __init__(self, use_lectures: bool = True, lecture_ratio: float = 0.3):
        """
        Args:
            use_lectures: Использовать ли лекции при обучении
            lecture_ratio: Доля примеров с контекстом из лекций (0.0-1.0)
        """
        self.path_config = PATH_CONFIG
        self.domain_config = DOMAIN_CONFIG
        self.question_counts = [2, 3, 5]
        self.use_lectures = use_lectures
        self.lecture_ratio = lecture_ratio
        
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
                                 num_examples: int,
                                 lectures: Optional[str] = None) -> List[Dict]:
        """
        Создает обучающие примеры из базы вопросов и лекций.
        
        Args:
            questions: Список вопросов
            domain_name: Название домена
            num_examples: Количество примеров
            lectures: Текст лекций (опционально)
            
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
        
        # Разбиваем лекции на чанки, если они есть
        lecture_chunks = []
        if lectures and self.use_lectures:
            lecture_chunks = split_text_into_chunks(lectures, max_length=400)
            logger.info(f"  Лекции разбиты на {len(lecture_chunks)} фрагментов")
        
        # Вычисляем количество примеров с/без лекций
        num_with_lectures = int(num_examples * self.lecture_ratio) if lecture_chunks else 0
        num_without_lectures = num_examples - num_with_lectures
        
        # СТРАТЕГИЯ 1: Примеры БЕЗ контекста из лекций (стандартные)
        for ticket_num in range(1, num_without_lectures + 1):
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
        
        # СТРАТЕГИЯ 2: Примеры С контекстом из лекций
        if lecture_chunks:
            for ticket_num in range(num_without_lectures + 1, num_examples + 1):
                num_questions = random.choice(self.question_counts)
                ticket = self.generate_ticket(
                    questions, num_questions, ticket_num, domain_name
                )
                
                # Извлекаем ключевые слова из билета
                keywords = extract_keywords(ticket, top_n=5)
                
                # Находим релевантные фрагменты лекций
                relevant_chunks = filter_relevant_chunks(lecture_chunks, keywords, top_k=2)
                
                if relevant_chunks:
                    context = "\n\n".join(relevant_chunks)
                    instruction = (
                        f"На основе следующего материала по {domain_name}:\n\n{context}\n\n"
                        f"Сгенерируй экзаменационный билет с {num_questions} вопросами:"
                    )
                else:
                    # Если релевантных чанков нет, используем случайный
                    random_chunk = random.choice(lecture_chunks)
                    instruction = (
                        f"Используя материал лекции по {domain_name}:\n\n{random_chunk}\n\n"
                        f"Составь экзаменационный билет с {num_questions} вопросами:"
                    )
                
                examples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": ticket
                })
        
        logger.info(f"  Создано {len(examples)} примеров (с лекциями: {num_with_lectures}, без: {num_without_lectures})")
        return examples

    def prepare_all_datasets(self):
        """Подготавливает датасеты для всех доменов."""
        print_section("ПОДГОТОВКА ОБУЧАЮЩИХ ДАТАСЕТОВ")
        
        if not self.path_config.train_data_dir.exists():
            raise FileNotFoundError(
                f"Папка {self.path_config.train_data_dir} не найдена!"
            )
        
        logger.info(f"Режим работы: {'С ЛЕКЦИЯМИ' if self.use_lectures else 'ТОЛЬКО ВОПРОСЫ'}")
        if self.use_lectures:
            logger.info(f"Доля примеров с лекциями: {self.lecture_ratio * 100}%\n")
        
        for domain_key, domain_info in self.domain_config.domains.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Обработка домена: {domain_key}")
            logger.info(f"{'='*60}")
            
            # Новая структура: domain/questions/questions.txt
            domain_dir = self.path_config.train_data_dir / domain_key
            questions_file = domain_dir / "questions" / "questions.txt"
            
            if not questions_file.exists():
                # Fallback на старую структуру
                questions_file = self._find_domain_file(domain_key, domain_info)
                if not questions_file:
                    logger.warning(
                        f"Файл для '{domain_key}' не найден. "
                        f"Ожидается: {domain_dir}/questions/questions.txt"
                    )
                    continue
            
            # Загрузка и валидация вопросов
            questions = load_questions_from_file(questions_file)
            questions = validate_questions(questions)
            
            if len(questions) < 10:
                logger.warning(
                    f"Мало вопросов для {domain_key}: {len(questions)}. "
                    "Рекомендуется 50+"
                )
            
            # Загрузка лекций, если нужно
            lectures = None
            if self.use_lectures:
                lectures = load_lectures_for_domain(domain_dir)
                if not lectures:
                    logger.info(f"  ⚠ Лекции не найдены, используем только вопросы")
            
            # Генерация примеров
            logger.info(f"Генерация {self.domain_config.num_examples_per_domain} примеров...")
            examples = self.create_training_examples(
                questions,
                domain_info['display_name'],
                self.domain_config.num_examples_per_domain,
                lectures
            )
            
            # Сохранение
            output_file = self.path_config.datasets_dir / f"{domain_key}_train.jsonl"
            save_jsonl(examples, output_file)
            logger.info(
                f"  ✓ Датасет: {len(examples)} примеров "
                f"из {len(questions)} вопросов"
            )
        
        print_section("ПОДГОТОВКА ЗАВЕРШЕНА")

    def _find_domain_file(self, domain_key: str, domain_info: Dict) -> Path:
        """Ищет файл с вопросами для домена (старая структура)."""
        for pattern in domain_info['file_patterns']:
            for ext in ['.txt', '.json', '.jsonl']:
                filepath = self.path_config.train_data_dir / f"{pattern}{ext}"
                if filepath.exists():
                    return filepath
        return None
