import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print("1. Загружаем данные...")
# Заметь, путь начинается с '../', потому что наш ноутбук лежит внутри папки notebooks/
train = pd.read_csv('data/raw/application_train.csv')
bureau = pd.read_csv('data/raw/bureau.csv')

print(f"Размер train: {train.shape}")
print(f"Размер bureau: {bureau.shape}")

print("\n2. Начинаем Feature Engineering (БКИ)...")
# Группируем прошлые кредиты по ID клиента
bureau_agg = bureau.groupby('SK_ID_CURR', as_index=False).agg({
    'SK_ID_BUREAU': 'count',       # Сколько всего кредитов было
    'DAYS_CREDIT': ['mean', 'min'],# Как давно брал кредиты (в днях)
    'AMT_CREDIT_SUM': 'sum',       # Общая сумма, которую он брал
    'AMT_CREDIT_SUM_DEBT': 'sum'   # Сколько он должен прямо сейчас
})

# Pandas после agg создает "двухэтажные" названия колонок. Схлопываем их в один этаж:
bureau_agg.columns =[
    'SK_ID_CURR', 
    'BUREAU_LOAN_COUNT', 
    'BUREAU_DAYS_CREDIT_MEAN', 
    'BUREAU_DAYS_CREDIT_MIN', 
    'BUREAU_AMT_CREDIT_SUM', 
    'BUREAU_AMT_CREDIT_SUM_DEBT'
]

print("Новые фичи созданы!")
bureau_agg.head()



print("3. Объединяем (Merge) новые фичи с главной таблицей...")
# Приклеиваем к train новые колонки по ID клиента (как LEFT JOIN в SQL)
train_merged = train.merge(bureau_agg, on='SK_ID_CURR', how='left')

print(f"Новый размер train: {train_merged.shape}")

print("\n4. Готовим данные и обучаем модель...")
X = train_merged.drop(columns=['SK_ID_CURR', 'TARGET'])
y = train_merged['TARGET']

# Обрабатываем категории
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category')

# Пропуски в числах заполняем медианой
num_cols = X.select_dtypes(exclude=['category']).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

# Сплит
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Учим LightGBM
model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=50)])

# Проверяем метрику
y_pred_proba = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_proba)
print(f"\n🚀 НОВЫЙ ROC AUC: {auc:.4f}")