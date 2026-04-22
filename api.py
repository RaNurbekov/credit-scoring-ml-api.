from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from src.database import init_db, log_request


ml_models = {}
schema_info = {} # 🧠 Память сервера для типов колонок

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Загружаю модель в память...")
    ml_models["lgbm"] = joblib.load('models/lgbm_baseline.pkl')
    
    # ИНЖЕНЕРНЫЙ ТРЮК: Читаем 1 строку обучающей выборки, чтобы запомнить схему
    print("⏳ Изучаю схему данных...")
    df_schema = pd.read_csv('models/schema.csv')
    # Запоминаем, какие колонки были текстовыми
    schema_info["cat_cols"] = df_schema.select_dtypes(include=['object']).columns.tolist()
    # Запоминаем правильный порядок колонок из самой модели!
    schema_info["all_features"] = ml_models["lgbm"].feature_name_

     # --- НОВАЯ СТРОЧКА ---
    init_db()
    
    
    print("✅ Модель и схема данных успешно загружены!")
    yield
    ml_models.clear()

app = FastAPI(title="Kaspi/Halyk Credit Scoring API", lifespan=lifespan)

class ClientData(BaseModel):
    features: dict

@app.post("/predict")
def predict(data: ClientData):
    df = pd.DataFrame([data.features])
    
    # 1. Выстраиваем колонки строго в том порядке, в котором модель обучалась
    df = df[schema_info["all_features"]]
    
    # 2. ЖЕСТКО контролируем типы данных (спасает от ошибки LightGBM)
    for col in df.columns:
        if col in schema_info["cat_cols"]:
            df[col] = df[col].astype('category')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 3. Прогноз
    prob = ml_models["lgbm"].predict_proba(df)[:, 1][0]
    
    # 4. Бизнес-логика (отказ при риске > 15%)
    decision = "Одобрить" if prob < 0.15 else "Отказать"

    # --- НОВАЯ СТРОЧКА: Сохраняем в БД ---
    log_request(data.features, float(prob), decision)
    
    return {
        "probability_of_default": round(float(prob), 4),
        "decision": decision
    }


