import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os

os.makedirs('models', exist_ok=True)

print("1. Загрузка данных...")
df = pd.read_csv('data/raw/application_train.csv')

print("2. Подготовка данных...")
X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
y = df['TARGET']

for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category')

num_cols = X.select_dtypes(exclude=['category']).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

print("3. Разделение на train/val...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("4. Обучение LightGBM...")
model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=50)])

print("5. Оценка модели...")
y_pred_proba = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_proba)
print(f"\n🎉 Validation ROC AUC: {auc:.4f}")

print("6. Сохранение модели...")
joblib.dump(model, 'models/lgbm_baseline.pkl')
print("Модель успешно сохранена в models/lgbm_baseline.pkl")