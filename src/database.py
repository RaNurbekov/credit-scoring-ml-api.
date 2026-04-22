import sqlite3
import json
from datetime import datetime
import os

# База данных будет храниться в файле logs.db прямо в корне проекта
DB_PATH = "logs.db"

def init_db():
    """Создает таблицу в базе данных, если её еще нет."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Пишем классический SQL запрос на создание таблицы
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_time TEXT,
            client_data TEXT,
            probability_of_default REAL,
            decision TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("🗄️ База данных SQLite успешно инициализирована!")

def log_request(features_dict: dict, prob: float, decision: str):
    """Записывает каждый запрос от пользователя в БД."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Получаем текущее время
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Сохраняем словарь с фичами в виде текста (JSON)
    features_json = json.dumps(features_dict)
    
    # SQL запрос на вставку данных
    cursor.execute('''
        INSERT INTO api_logs (request_time, client_data, probability_of_default, decision)
        VALUES (?, ?, ?, ?)
    ''', (now, features_json, prob, decision))
    
    conn.commit()
    conn.close()