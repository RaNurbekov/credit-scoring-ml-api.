# 1. Берем за основу легкую версию Linux с установленным Python 3.10
FROM python:3.10-slim

# 2. ИНЖЕНЕРНЫЙ СЕКРЕТ: Для LightGBM в Linux нужна системная библиотека libgomp1
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# 3. Устанавливаем рабочую папку внутри контейнера
WORKDIR /app

# 4. Копируем файл с библиотеками и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Копируем сам код API, папку с моделью и папку data
COPY api.py .
COPY models/ models/
COPY data/ data/

# 6. Открываем порт 8000
EXPOSE 8000

# 7. Запускаем сервер (тут был пропущен пробел после CMD!)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
