import pandas as pd
import requests
import json

print("1. Читаем анкету первого клиента из тестовой базы...")
df_test = pd.read_csv('data/raw/application_test.csv', nrows=1)

# Убираем ID
df_test = df_test.drop(columns=['SK_ID_CURR'])

# ИСПРАВЛЕНИЕ ЗДЕСЬ: to_json() правильно превращает пустоты (NaN) в null для JSON
client_json_str = df_test.iloc[0].to_json()
client_dict = json.loads(client_json_str)

print("2. Отправляем запрос на наш работающий сервер (API)...")
url = "https://credit-scoring-ml-api.onrender.com/predict"
payload = {"features": client_dict}

# Делаем POST запрос
response = requests.post(url, json=payload)

# Проверяем, всё ли хорошо на стороне сервера (код 200 = ОК)
if response.status_code == 200:
    print("\n=== ОТВЕТ ОТ СЕРВЕРА БАНКА ===")
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))
else:
    print(f"\n❌ ОШИБКА СЕРВЕРА (Код: {response.status_code}):")
    print(response.text)