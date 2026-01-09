import json
import logging
from pathlib import Path
from typing import List, Dict

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('exam_generator.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def load_questions_from_file(filepath: Path) -> List[str]:
    """
    Загружает вопросы из файла.
    
    Поддерживает форматы:
    - .txt: вопросы разделены пустыми строками или построчно
    - .json: список вопросов или объект с полем 'questions'
    - .jsonl: по вопросу в строке
    
    Args:
        filepath: Путь к файлу с вопросами
    
    Returns:
        Список вопросов
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Файл {filepath} не найден")
    
    questions = []
    
    try:
        if filepath.suffix == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Разделение по двойным переносам
            parts = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            if len(parts) > 1:
                questions = parts
            else:
                # Разделение по одиночным переносам
                questions = [
                    line.strip() for line in content.split('\n')
                    if line.strip() and not line.startswith('#')
                ]
        
        elif filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                questions = data
            elif isinstance(data, dict) and 'questions' in data:
                questions = data['questions']
            else:
                raise ValueError("JSON должен содержать список или поле 'questions'")
        
        elif filepath.suffix == '.jsonl':
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if isinstance(data, str):
                            questions.append(data)
                        elif isinstance(data, dict) and 'question' in data:
                            questions.append(data['question'])
        else:
            raise ValueError(f"Неподдерживаемый формат: {filepath.suffix}")
        
        logger.info(f"Загружено {len(questions)} вопросов из {filepath.name}")
        return questions
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке {filepath}: {e}")
        raise

def validate_questions(questions: List[str], min_length: int = 10) -> List[str]:
    """
    Валидирует и очищает список вопросов.
    
    Args:
        questions: Список вопросов
        min_length: Минимальная длина вопроса
    
    Returns:
        Очищенный список вопросов
    """
    valid_questions = []
    
    for q in questions:
        q = ' '.join(q.split())  # Убираем лишние пробелы
        
        if len(q) >= min_length and q not in valid_questions:
            valid_questions.append(q)
    
    logger.info(f"Валидация: {len(questions)} -> {len(valid_questions)} вопросов")
    return valid_questions

def save_jsonl(data: List[Dict], filepath: Path):
    """Сохраняет данные в JSONL формате."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Сохранено {len(data)} записей в {filepath}")

def print_section(title: str, width: int = 80, char: str = '='):
    """Выводит форматированный заголовок раздела."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")

def print_progress(current: int, total: int, prefix: str = "Прогресс"):
    """Выводит прогресс-бар."""
    percent = 100 * (current / float(total))
    filled = int(50 * current // total)
    bar = '█' * filled + '-' * (50 - filled)
    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})', end='')
    if current == total:
        print()
