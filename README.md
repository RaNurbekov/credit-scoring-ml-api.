# 🏦 Credit Scoring ML API (with MLOps)

Микросервис машинного обучения для оценки кредитного риска (вероятности дефолта) клиентов. Проект демонстрирует полный цикл ML Engineering: от обучения модели до трекинга экспериментов и динамического деплоя REST API в Docker-контейнере.

## 🛠 Стек технологий
* **Machine Learning:** Python, Pandas, Scikit-Learn, LightGBM
* **Backend:** FastAPI, Uvicorn, Pydantic
* **DevOps & MLOps:** MLflow (Model Registry & Experiment Tracking), Docker, Git
* **Explainable AI:** SHAP (Интерпретация решений модели)

## ⚙️ Архитектура проекта
1. **Experiment Tracking (`src/train.py`):** Обучение модели на данных [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk). Интегрирован **MLflow** с функцией `autolog()` для автоматического сохранения гиперпараметров, метрик (ROC-AUC) и артефактов в локальную БД SQLite.
2. **Dynamic Loading API (`api.py`):** REST API на FastAPI. Модель не "зашита" в код жестко — при старте сервер динамически скачивает нужную версию модели напрямую из базы данных MLflow по `RUN_ID`.
3. **Data Validation:** Реализован жесткий контроль схемы данных для защиты от дрейфа типов (Data Drift).
4. **Containerization (`Dockerfile`):** Упаковка сервиса в Docker-контейнер для изоляции окружения.

## 🚀 Как запустить проект локально

### 1. Подготовка данных
Скачайте датасет[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) с Kaggle и поместите `application_train.csv` в папку `data/raw/`.

### 2. Обучение и Трекинг (MLflow)
Запустите скрипт обучения. MLflow автоматически запишет результаты в `mlflow.db`:
```bash
python src/train.py


## Для просмотра дашборда экспериментов запустите сервер:

mlflow server --host 127.0.0.1 --port 5000 --workers 1 --backend-store-uri sqlite:///mlflow.db

## 3. Запуск API
 #Скопируйте RUN_ID лучшей модели из дашборда MLflow, вставьте его в api.py и запустите сервер:

uvicorn api:app --reload