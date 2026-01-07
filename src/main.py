"""
Главная точка входа для системы генерации экзаменационных билетов.
"""

import sys
import argparse
from pathlib import Path

# Добавляем корневую директорию в путь для импортов
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DatasetGenerator
from src.model import AdapterTrainer
from src.generation import ExamTicketGenerator
from src.utils import print_section, logger
from src.core import PATH_CONFIG, DOMAIN_CONFIG

def setup_example_data():
    """Создает примеры файлов с вопросами."""
    train_data_dir = PATH_CONFIG.train_data_dir
    train_data_dir.mkdir(parents=True, exist_ok=True)
    
    examples = {
        "cryptography.txt": [
            "Объясните принцип работы алгоритма RSA и приведите математическое обоснование процедур генерации ключей, шифрования и расшифрования.",
            "Опишите режимы работы блочных шифров (ECB, CBC, CTR, GCM). В чем заключаются преимущества и недостатки каждого режима?",
            "Что такое криптографическая хеш-функция и каким требованиям она должна удовлетворять? Объясните атаки на хеш-функции.",
            "Опишите протокол обмена ключами Диффи-Хеллмана и его уязвимость к атаке 'человек посередине'.",
            "Объясните принципы работы эллиптических кривых в криптографии (ECC).",
            "Что такое цифровая подпись? Опишите схемы RSA и DSA для цифровой подписи.",
            "Объясните концепцию квантовых вычислений и их влияние на современную криптографию.",
            "Что такое блочный шифр? Опишите структуру сети Фейстеля.",
            "Объясните принцип работы алгоритма AES. Какие операции выполняются в каждом раунде?",
            "Что такое постквантовая криптография и какие подходы в ней используются?"
        ],
        "networks.txt": [
            "Опишите модель OSI и стек протоколов TCP/IP. Какие функции выполняет каждый уровень?",
            "Объясните принцип работы протокола TCP. Как обеспечивается надежная доставка данных?",
            "Что такое IP-адресация и маска подсети? Объясните различие между IPv4 и IPv6.",
            "Опишите алгоритмы маршрутизации: дистанционно-векторные и алгоритмы состояния канала.",
            "Что такое NAT (Network Address Translation) и для чего он используется?",
            "Объясните процесс установления TCP-соединения (three-way handshake).",
            "Что такое DNS? Опишите процесс разрешения доменного имени.",
            "Объясните различия между коммутатором и маршрутизатором.",
            "Что такое VLAN и для чего используется сегментация сети?",
            "Опишите протокол HTTPS и процесс установления защищенного соединения."
        ],
        "algorithms.txt": [
            "Опишите структуру данных 'красно-черное дерево'. Какие инварианты должны соблюдаться?",
            "Объясните принцип работы алгоритма быстрой сортировки (QuickSort) и его временную сложность.",
            "Что такое динамическое программирование? Приведите пример классической задачи.",
            "Опишите алгоритм поиска в ширину (BFS) и его применение для поиска кратчайшего пути.",
            "Что такое хеш-таблица? Как разрешаются коллизии методом цепочек и открытой адресацией?",
            "Объясните алгоритм Дейкстры для поиска кратчайших путей в графе.",
            "Что такое сбалансированное бинарное дерево поиска? Приведите примеры (AVL, красно-черное).",
            "Опишите жадные алгоритмы и приведите примеры их применения (алгоритм Прима, Краскала).",
            "Объясните алгоритм сортировки слиянием (Merge Sort) и его временную сложность.",
            "Что такое NP-полные задачи? Приведите примеры и объясните их значимость."
        ]
    }
    
    for filename, questions in examples.items():
        filepath = train_data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(questions))
    
    logger.info(f"✓ Созданы примеры файлов в {train_data_dir}")

def prepare_data():
    """Этап 1: Подготовка датасетов."""
    print_section("ЭТАП 1: ПОДГОТОВКА ДАТАСЕТОВ")
    generator = DatasetGenerator()
    generator.prepare_all_datasets()

def train_adapters():
    """Этап 2: Обучение адаптеров."""
    print_section("ЭТАП 2: ОБУЧЕНИЕ АДАПТЕРОВ")
    trainer = AdapterTrainer()
    trainer.train_all_adapters()

def generate_tickets(domain: str, num_questions: int, num_tickets: int):
    """Этап 3: Генерация билетов."""
    print_section("ЭТАП 3: ГЕНЕРАЦИЯ БИЛЕТОВ")
    
    generator = ExamTicketGenerator()
    
    tickets = generator.generate_multiple_tickets(
        domain_key=domain,
        num_questions=num_questions,
        num_tickets=num_tickets
    )
    
    # Вывод примера
    print("\n" + "="*60)
    print("ПРИМЕР БИЛЕТА:")
    print("="*60)
    print(tickets[0])
    print("="*60)
    
    # Сохранение
    filepath = generator.save_tickets(tickets, domain)
    print(f"\n✓ Все билеты сохранены: {filepath}")

def main():
    """Основная функция с CLI."""
    parser = argparse.ArgumentParser(
        description="Система генерации экзаменационных билетов (Mistral 7B + LoRA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s --setup-example --mode full
  %(prog)s --mode prepare
  %(prog)s --mode train
  %(prog)s --mode generate --domain crypto --num-questions 5 --num-tickets 10
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['prepare', 'train', 'generate', 'full'],
        default='full',
        help='Режим: prepare, train, generate, full (все этапы)'
    )
    
    parser.add_argument(
        '--domain',
        choices=['crypto', 'networks', 'algorithms'],
        default='crypto',
        help='Предметная область для генерации'
    )
    
    parser.add_argument(
        '--num-questions',
        type=int,
        default=3,
        choices=[2, 3, 5],
        help='Количество вопросов в билете'
    )
    
    parser.add_argument(
        '--num-tickets',
        type=int,
        default=5,
        help='Количество билетов'
    )
    
    parser.add_argument(
        '--setup-example',
        action='store_true',
        help='Создать примеры файлов с вопросами'
    )
    
    args = parser.parse_args()
    
    print_section("СИСТЕМА ГЕНЕРАЦИИ ЭКЗАМЕНАЦИОННЫХ БИЛЕТОВ")
    print("Mistral 7B + LoRA Fine-tuning")
    print(f"Проект: {PROJECT_ROOT}\n")
    
    # Создание примеров
    if args.setup_example:
        setup_example_data()
    
    # Выполнение
    try:
        if args.mode in ['prepare', 'full']:
            prepare_data()
        
        if args.mode in ['train', 'full']:
            train_adapters()
        
        if args.mode in ['generate', 'full']:
            generate_tickets(args.domain, args.num_questions, args.num_tickets)
        
        print_section("✓ ВЫПОЛНЕНО УСПЕШНО")
    
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        print_section("✗ ОШИБКА ВЫПОЛНЕНИЯ")
        sys.exit(1)

if __name__ == "__main__":
    main()
