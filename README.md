# 🏦 Credit Scoring ML API
Live API: https://credit-scoring-ml-api.onrender.com/predict

Микросервис машинного обучения для оценки кредитного риска (вероятности дефолта) клиентов. Проект демонстрирует полный цикл ML Engineering: от обучения модели до деплоя REST API в Docker-контейнере.

## 🛠 Стек технологий
* **Machine Learning:** Python, Pandas, Scikit-Learn, LightGBM
* **Backend:** FastAPI, Uvicorn, Pydantic
* **DevOps / MLOps:** Docker, Git

## ⚙️ Архитектура проекта
1. **Model Training (`src/train.py`):** Обучение модели `LightGBM` на данных соревнования[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).
2. **API Service (`api.py`):** REST API на FastAPI, которое принимает JSON с признаками клиента и возвращает вероятность дефолта и бизнес-решение (Одобрить/Отказать). Реализован жесткий контроль схемы данных для защиты от дрейфа типов (Data Drift).
3. **Containerization (`Dockerfile`):** Упаковка сервиса в Docker-контейнер для изоляции окружения и удобного деплоя.

## 🚀 Как запустить проект локально

### 1. Подготовка данных
Скачайте датасет [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) с Kaggle и поместите файлы в папку `data/raw/`.

### 2. Обучение модели
Перед запуском API необходимо обучить модель (файл `.pkl` не включен в репозиторий из-за ограничений по размеру):
```bash
python src/train.py
