# 🏦 Credit Scoring ML API (with Advanced MLOps)

![CI Pipeline](https://github.com/RaNurbekov/credit-scoring-ml-api/actions/workflows/ci-pipeline.yml/badge.svg)

Микросервис машинного обучения для оценки кредитного риска. Проект демонстрирует **Advanced MLOps**: от трекинга экспериментов и динамического деплоя до мониторинга дрейфа данных (Data Drift) и автоматического CI/CD пайплайна.

## 🛠 Стек технологий
* **Machine Learning:** LightGBM, Pandas, Scikit-Learn
* **Backend:** FastAPI, Uvicorn, Pydantic
* **MLOps & DevOps:** MLflow, Docker, Git
* **CI/CD & Testing:** GitHub Actions, Pytest
* **Model Monitoring:** Evidently AI
* **Explainable AI:** SHAP

## ⚙️ Архитектура и MLOps процессы
1. **Experiment Tracking (`src/train.py`):** Интегрирован **MLflow** (`autolog`) для сохранения метрик и артефактов в локальную БД SQLite.
2. **Dynamic Loading API (`api.py`):** Сервер скачивает веса модели "на лету" из базы MLflow по `RUN_ID`, избавляя от жесткой привязки файлов.
3. **Continuous Integration (CI):** При каждом `git push` сервер **GitHub Actions** поднимает чистое окружение Linux и прогоняет Unit-тесты (`pytest`) для проверки бизнес-логики банка.
4. **Data Drift Monitoring (`src/monitor_drift.py`):** Скрипт на базе **Evidently AI**, генерирующий интерактивный HTML-дашборд. Позволяет вовремя отследить деградацию модели из-за инфляции или изменения профиля клиентов.

## 🚀 Как запустить проект

### 1. Обучение и MLflow
```bash
python src/train.py
mlflow server --host 127.0.0.1 --port 5000 --workers 1 --backend-store-uri sqlite:///mlflow.db

2. Запуск API
Скопируйте RUN_ID из дашборда MLflow, вставьте его в api.py и запустите сервер:
uvicorn api:app --reload

3. Мониторинг дрейфа данных (Evidently AI)
Сгенерируйте HTML-отчет, анализирующий сдвиги в данных:

python src/monitor_drift.py

4. Запуск Unit-тестов локально
python -m pytest -p no:langsmith tests/

