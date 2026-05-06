import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

print("1. Загружаем исторические данные (Reference) и новые данные (Current)...")
# Берем по 5000 строк для скорости
reference_data = pd.read_csv('data/raw/application_train.csv', nrows=5000)
current_data = pd.read_csv('data/raw/application_test.csv', nrows=5000)

print("2. 💥 Имитируем жесткий Data Drift (Кризис и Инфляция)...")
# Представим, что прошел год, и доходы клиентов (и суммы кредитов) выросли в 3 раза!
current_data['AMT_INCOME_TOTAL'] = current_data['AMT_INCOME_TOTAL'] * 3.0
current_data['AMT_CREDIT'] = current_data['AMT_CREDIT'] * 3.0

# Чтобы отчет сформировался быстро и красиво, выберем 5 самых важных числовых колонок
features_to_monitor =[
    'AMT_INCOME_TOTAL', 
    'AMT_CREDIT', 
    'DAYS_BIRTH', 
    'DAYS_EMPLOYED',
    'REGION_POPULATION_RELATIVE'
]

ref_df = reference_data[features_to_monitor].fillna(0)
curr_df = current_data[features_to_monitor].fillna(0)

print("3. Запускаем анализатор Evidently AI...")
# Создаем отчет, который проверит только Дрейф Данных
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=ref_df, current_data=curr_df)

print("4. Генерируем HTML-дашборд...")
os.makedirs('reports', exist_ok=True)
drift_report.save_html('reports/data_drift_report.html')

print("✅ Готово! Откройте файл reports/data_drift_report.html в любом браузере!")