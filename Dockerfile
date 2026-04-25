# 1. Берем за основу легкую версию Linux с установленным Python 3.10
FROM python:3.10-slim

# 2.  LightGBM в Linux нужна системная библиотека libgomp1
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY api.py .
COPY models/ models/
COPY src/ src/

EXPOSE 8000


CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
