import sqlite3
import pandas as pd

print("Подключаемся к базе данных банка...")
conn = sqlite3.connect('logs.db')

# Пишем классический SQL-запрос. 
# Мы не берем колонку client_data, чтобы огромный JSON не сломал нам красивый вывод на экран
sql_query = """
    SELECT id, request_time, probability_of_default, decision 
    FROM api_logs
"""

# Магия Pandas: он умеет сам выполнять SQL-запросы и сразу превращать результат в красивую таблицу!
df_logs = pd.read_sql_query(sql_query, conn)

print("\n=== ИСТОРИЯ ЗАПРОСОВ В БАНК ===")
print(df_logs)

# Как настоящие инженеры, не забываем закрывать соединение
conn.close()