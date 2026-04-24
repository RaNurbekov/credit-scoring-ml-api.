import streamlit as st
import requests
import pandas as pd

# 1. Настройка страницы
st.set_page_config(page_title="Kaspi ML API", page_icon="🏦", layout="centered")
st.title("🏦 Кредитный Скоринг (Demo)")
st.write("Этот интерфейс отправляет данные на облачный ML-микросервис и предсказывает вероятность дефолта.")

# Твоя ссылка на облако Render!
API_URL = "https://credit-scoring-ml-api.onrender.com/predict"

# 2. Функция для загрузки "скелета" клиента
@st.cache_data
def load_base_client():
    # Берем одну строчку из нашей схемы
    df = pd.read_csv("models/schema.csv")
    if 'TARGET' in df.columns:
        df = df.drop(columns=['TARGET'])
    if 'SK_ID_CURR' in df.columns:
        df = df.drop(columns=['SK_ID_CURR'])
    return df.iloc[0].fillna("").to_dict()

base_features = load_base_client()

# 3. Рисуем боковую панель с ползунками (UI)
st.sidebar.header("Параметры клиента")

age = st.sidebar.slider("Возраст (лет)", 18, 80, 35)
days_employed = st.sidebar.slider("Стаж работы (лет)", 0, 40, 5)
income = st.sidebar.number_input("Доход (в год)", 50000, 10000000, 150000, step=10000)
credit_amt = st.sidebar.number_input("Сумма кредита", 10000, 5000000, 500000, step=50000)

# --- НОВЫЙ БЛОК ---
st.sidebar.markdown("---")
st.sidebar.subheader("Скрытые рейтинги (БКИ)")
ext_2 = st.sidebar.slider("Внешний рейтинг 2", 0.0, 1.0, 0.6)
ext_3 = st.sidebar.slider("Внешний рейтинг 3", 0.0, 1.0, 0.6)

# --- НОВЫЙ БЛОК: ИСТОРИЯ ПРОСРОЧЕК ---
st.sidebar.markdown("---")
st.sidebar.subheader("История платежей")
max_past_due = st.sidebar.slider("Максимальная просрочка (дней)", 0, 365, 0)
total_payments = st.sidebar.slider("Всего выплачено кредитов", 0, 50, 10)

# 
# В датасете возраст и стаж считаются в днях со знаком минус (особенность Home Credit)
# 4. Обновляем скелет новыми данными из ползунков
base_features['DAYS_BIRTH'] = -int(age * 365.25)
base_features['DAYS_EMPLOYED'] = -int(days_employed * 365.25)
base_features['AMT_INCOME_TOTAL'] = float(income)
base_features['AMT_CREDIT'] = float(credit_amt)

# --- НОВЫЕ СТРОЧКИ ---
base_features['EXT_SOURCE_2'] = float(ext_2)
base_features['EXT_SOURCE_3'] = float(ext_3)

# --- НОВЫЕ СТРОЧКИ ---
base_features['MAX_PAST_DUE_DAYS'] = float(max_past_due)
# Для простоты считаем среднюю просрочку равной максимальной (если она есть)
base_features['MEAN_PAST_DUE_DAYS'] = float(max_past_due) / 2 if max_past_due > 0 else 0.0
base_features['TOTAL_PAYMENTS'] = float(total_payments)


# 5. Кнопка отправки в облако
if st.button("🚀 Рассчитать вероятность дефолта", use_container_width=True):
    with st.spinner("Связываюсь с дата-центром во Франкфурте... 🌍"):
        try:
            # Отправляем JSON на твой API
            response = requests.post(API_URL, json={"features": base_features})
            
            if response.status_code == 200:
                result = response.json()
                prob = result["probability_of_default"]
                decision = result["decision"]
                
                st.markdown("---")
                st.subheader("Вердикт модели:")
                
                # Красивый вывод результата
                if decision == "Одобрить":
                    st.success(f"✅ **{decision}** (Риск: {prob:.2%})")
                    st.balloons() # Магия Streamlit - шарики при успехе!
                else:
                    st.error(f"❌ **{decision}** (Риск: {prob:.2%})")
            else:
                st.error(f"Ошибка сервера: {response.status_code}")
        except Exception as e:
            st.error(f"Не удалось подключиться к API: {e}")