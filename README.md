# 🏦 Credit Risk API — Full MLOps Pipeline

> **Полный цикл ML Engineering для банковского кредитного скоринга:**
> Данные → Обучение → Трекинг → API → Мониторинг Дрейфа → Docker

🔗 **Live API:** https://credit-scoring-ml-api.onrender.com/predict

---

## 🛠 Стек технологий

| Слой | Технологии |
|---|---|
| **Machine Learning** | Python, Pandas, Scikit-Learn, LightGBM |
| **Experiment Tracking** | MLflow (autolog, Model Registry, SQLite backend) |
| **Explainable AI** | SHAP (TreeExplainer, Top-5 факторов риска) |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Мониторинг** | Evidently AI (Data Drift Detection) |
| **Frontend** | Streamlit (интерактивный UI для скоринга) |
| **DevOps** | Docker, Git |
| **Логирование** | SQLite (история всех предсказаний) |

---

## ⚙️ Архитектура системы

```
📂 Home Credit Dataset (Kaggle)
        │
        ▼
🔬 src/train.py  ──► MLflow autolog() ──► mlflow.db (SQLite)
        │                                      │
        │                               (метрики, ROC-AUC,
        │                                гиперпараметры, артефакты)
        ▼
🚀 api.py (FastAPI)
        │
        ├── lifespan: mlflow.lightgbm.load_model(RUN_ID) ──► динамическая загрузка
        │
        ├── /predict ──► SHAP TreeExplainer ──► Топ-5 факторов решения
        │            └──► log_request() ──► SQLite (аудит-лог)
        │
        └── Streamlit UI (app.py) ──► визуальный скоринг-дашборд
        
📊 src/monitor_drift.py ──► Evidently AI ──► reports/data_drift_report.html
```

---

## 🔑 Ключевые особенности

### 1. MLflow Model Registry
Модель не "зашита" в код — при старте сервер **динамически загружает** нужную версию из MLflow по `RUN_ID`. Это позволяет переключаться между версиями модели без изменения кода API:
```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
ml_models["lgbm"] = mlflow.lightgbm.load_model(f"runs:/{RUN_ID}/model")
```

### 2. Explainable AI (SHAP)
Каждое решение по кредиту сопровождается **объяснением** — топ-5 факторов которые повлияли на результат. Это требование банковских регуляторов (BASEL III):
```json
{
  "probability_of_default": 0.73,
  "decision": "Отказать",
  "explanation": [
    {"feature": "AMT_CREDIT", "impact": +0.42},
    {"feature": "DAYS_EMPLOYED", "impact": +0.31},
    ...
  ]
}
```

### 3. Data Drift Monitoring (Evidently AI)
`src/monitor_drift.py` сравнивает референсные и текущие данные по 5 ключевым фичам модели и генерирует HTML-дашборд с алертами о дрейфе. Симулируется сценарий кризиса (рост доходов и кредитов в 3x):
```bash
python src/monitor_drift.py
# → reports/data_drift_report.html
```

### 4. Аудит-лог всех предсказаний
Каждый `/predict` запрос логируется в SQLite: фичи клиента, вероятность дефолта, решение, timestamp. Полная воспроизводимость и аудируемость.

---

## 🚀 Как запустить проект

### 1. Подготовка данных
Скачайте датасет [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) с Kaggle и поместите `application_train.csv` и `application_test.csv` в папку `data/raw/`.

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Обучение модели + MLflow трекинг
```bash
python src/train.py
```
MLflow автоматически сохранит гиперпараметры, метрики (ROC-AUC) и артефакты модели в `mlflow.db`.

### 4. Просмотр экспериментов в MLflow UI
```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```
Откройте http://localhost:5000, скопируйте `RUN_ID` лучшей модели и вставьте в `api.py`.

### 5. Запуск API
```bash
# Локально
uvicorn api:app --reload

# Через Docker
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

### 6. Запуск Streamlit UI
```bash
streamlit run app.py
```

### 7. Мониторинг Data Drift
```bash
python src/monitor_drift.py
# Откройте reports/data_drift_report.html в браузере
```

---

## 📁 Структура проекта

```
credit-risk-api/
├── src/
│   ├── train.py              # Обучение + MLflow autolog
│   ├── database.py           # SQLite аудит-лог предсказаний
│   └── monitor_drift.py      # Evidently AI Data Drift
├── models/                   # Сохранённые артефакты
├── notebooks/                # EDA и эксперименты
├── reports/                  # HTML-отчёты Evidently
├── api.py                    # FastAPI микросервис
├── app.py                    # Streamlit UI
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📊 Метрики модели

| Метрика | Значение |
|---|---|
| ROC-AUC | логируется в MLflow |
| Датасет | Home Credit Default Risk (Kaggle) |
| Алгоритм | LightGBM + Class Imbalance handling |
| Порог решения | 0.15 (настраивается) |

---

## 🔗 Ресурсы

- [Датасет на Kaggle](https://www.kaggle.com/c/home-credit-default-risk)
- [MLflow документация](https://mlflow.org/docs/latest/index.html)
- [Evidently AI документация](https://docs.evidentlyai.com/)
- [SHAP документация](https://shap.readthedocs.io/)
