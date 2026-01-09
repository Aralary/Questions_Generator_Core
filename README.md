# Questions_Generator_Core

В этом репозитории находится программа для дообучения и таргетирования модели

## Инструкция к использованию

### Создание и активация виртуального окружения
Linux
```
python -m venv venv
source venv/bin/activate
```

Windows
```
python.exe -m venv venv
.\venv\Scripts\activate
```

### Установка зависимостей
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

### Варианты запуска
1) Поэтапно:
- Этап подготовки датасетов
```
python src/main.py --mode prepare
```

- Этап обучения адаптеров
```
python src/main.py --mode train
```

- Этап генерации билетов

```
Криптография, 5 вопросов, 10 билетов:
python src/main.py --mode generate --domain cryptography --num-questions 5 --num-tickets 10

Компьютерные сети, 3 вопроса, 5 билетов:
python src/main.py --mode generate --domain networks --num-questions 3 --num-tickets 5

Алгоритмы, 2 вопроса, 8 билетов:
python src/main.py --mode generate --domain algorithms --num-questions 2 --num-tickets 8
```

2) Запуск всех этапов одной командой
```
python src/main.py --mode full --domain cryptography --num-questions 3 --num-tickets 5
```