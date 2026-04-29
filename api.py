from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow # <-- НОВЫЙ ИМПОРТ

ml_models = {}
schema_info = {}

# ВСТАВЬ СЮДА СВОЙ СКОПИРОВАННЫЙ RUN ID!
RUN_ID = "a2111785183b4863ac88988d6f55965b"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Подключаюсь к реестру MLflow...")
    # Говорим API, где лежит наша база данных с моделями
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    print(f"⏳ Скачиваю модель из запуска: {RUN_ID}...")
    # Формируем специальный путь MLflow
    model_uri = f"runs:/{RUN_ID}/model"
    
    # Магия! Скачиваем модель прямо из базы MLflow в оперативную память!
    ml_models["lgbm"] = mlflow.lightgbm.load_model(model_uri)
    
    print("⏳ Изучаю схему данных...")
    df_schema = pd.read_csv('data/raw/application_train.csv', nrows=1)
    schema_info["cat_cols"] = df_schema.select_dtypes(include=['object']).columns.tolist()
    # LightGBM из MLflow немного иначе хранит имена фичей, достаем их безопасно
    schema_info["all_features"] = ml_models["lgbm"].feature_name_
    
    print("✅ Сервер готов! Модель загружена из MLflow.")
    yield
    ml_models.clear()

app = FastAPI(title="Kaspi/Halyk Credit Scoring API", lifespan=lifespan)

class ClientData(BaseModel):
    features: dict

@app.post("/predict")
def predict(data: ClientData):
    df = pd.DataFrame([data.features])
    df = df[schema_info["all_features"]]
    
    for col in df.columns:
        if col in schema_info["cat_cols"]:
            df[col] = df[col].astype('category')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    prob = ml_models["lgbm"].predict_proba(df)[:, 1][0]
    decision = "Одобрить" if prob < 0.15 else "Отказать"
    
    # --- НОВЫЙ БЛОК: Рентген модели ---
    # Считаем вклад каждой фичи для конкретно этого клиента
    shap_values = ml_models["explainer"].shap_values(df)
    
    # LightGBM возвращает список для 2 классов. Берем индекс [1] (Дефолт)
    client_shap = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    
    # Создаем словарь: {Название_колонки: Вклад_в_риск}
    shap_dict = {feat: float(val) for feat, val in zip(df.columns, client_shap)}
    
    # Сортируем фичи по их СИЛЕ (абсолютному значению), берем ТОП-5 самых важных
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_explanation =[{"feature": k, "impact": v} for k, v in sorted_shap[:5]]
    
    log_request(data.features, float(prob), decision)
    
    return {
        "probability_of_default": round(float(prob), 4),
        "decision": decision,
        "explanation": top_explanation # Отправляем объяснение на сайт!
    }