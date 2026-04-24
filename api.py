from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
from src.database import init_db, log_request

ml_models = {}
schema_info = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Загружаю модель и SHAP Explainer...")
    ml_models["lgbm"] = joblib.load('models/lgbm_baseline.pkl')
    
    # --- МАГИЯ SHAP ---
    # Создаем "объяснителя", который выучит логику нашей LightGBM модели
    ml_models["explainer"] = shap.TreeExplainer(ml_models["lgbm"])
    
    df_schema = pd.read_csv('models/schema.csv')
    schema_info["cat_cols"] = df_schema.select_dtypes(include=['object']).columns.tolist()
    schema_info["all_features"] = ml_models["lgbm"].feature_name_
    
    init_db()
    print("✅ Сервер полностью готов!")
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