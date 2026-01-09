# 🎓 Questions Generator Core

<div align="center">

**Система обучения LoRA-адаптеров для генерации экзаменационных вопросов с использованием Mistral-7B**

[![Python 3.12](https://img.shields.io/badge/python-3.12.0-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## 📖 Описание

**Questions_Generator_Core** — это модульная система для fine-tuning языковой модели **Mistral-7B-Instruct** методом **LoRA (Low-Rank Adaptation)** для генерации академических вопросов по специализированным предметным областям.

### 🎯 Назначение проекта

Проект предназначен для создания и обучения специализированных LoRA-адаптеров, которые затем используются в Telegram-боте **Questions_Generator_TGBot** для автоматической генерации экзаменационных билетов.

### 📚 Поддерживаемые предметные области

- **Криптография** — алгоритмы шифрования, цифровые подписи, протоколы безопасности
- **Компьютерные сети** — протоколы TCP/IP, маршрутизация, сетевая безопасность
- **Алгоритмы и структуры данных** — сортировки, графы, деревья, динамическое программирование

---

## ✨ Возможности

- 🧠 **Fine-tuning Mistral-7B** с использованием QLoRA (4-bit квантизация)
- 📄 **Поддержка лекций** — парсинг PDF/PPTX для контекстного обучения
- ⚡ **GPU Acceleration** — автоматическое использование NVIDIA GPU (если доступна)
- 🎲 **Гибкая генерация** — настройка количества вопросов и билетов
- 📊 **Модульная архитектура** — раздельные адаптеры для каждого домена
- 💾 **Эффективность** — адаптеры весят ~500MB (вместо 14GB полной модели)

---

## 💻 Системные требования

### Минимальные требования

| Компонент | Требование |
|-----------|------------|
| **Python** | 3.12.0 |
| **ОС** | Windows 10/11, Linux, macOS |
| **RAM** | 16 GB (32 GB рекомендуется) |
| **Диск** | 25 GB свободного места |

### Для обучения с GPU (рекомендуется)

| Компонент | Требование |
|-----------|------------|
| **GPU** | NVIDIA с поддержкой CUDA |
| **VRAM** | 12 GB+ (RTX 3060 12GB и выше) |
| **CUDA** | 12.1+ (совместимо с 13.0) |
| **Драйверы** | NVIDIA 525.60.11+ |

> ⚠️ **Примечание:** Обучение на CPU возможно, но займет в 10-15 раз больше времени (4-8 часов вместо 20-40 минут)

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/yourusername/Questions_Generator_Core.git
cd Questions_Generator_Core
```

### 2. Создание и активация виртуального окружения
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

### 3. Установка зависимостей

С GPU (CUDA 12.4+):
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

CPU-only:
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
### 4. Данные для обучения

Поместите вопросы в ***questions.txt***. Каждый вопрос в одной строке и отделен от предыдущего пустой строкой
```
data/train_data/
├── cryptography/questions/questions.txt
├── algorithms/questions/questions.txt
└── networks/questions/questions.txt
```

Поместите лекции в формате pdf или pptx в соответствующие директории ***lectures***
```
data/train_data/
├── cryptography/lectures/
├── algorithms/lectures/
└── networks/lectures/
```

### 5. Варианты запуска

1) Поэтапно:

- Этап подготовки датасетов
```
python src/main.py --mode prepare --use-lectures --lecture-ratio 0.5

или без использования лекций:

python src/main.py --mode prepare --no-lectures
```

- Этап создания и обучения адаптеров
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