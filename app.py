import streamlit as st
import requests
import pandas as pd

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
                
                # Используем get на случай, если API еще не успело обновиться
                explanation = result.get("explanation",[])
                
                st.markdown("---")
                st.subheader("Вердикт модели:")
                
                if decision == "Одобрить":
                    st.success(f"✅ **{decision}** (Риск: {prob:.2%})")
                    st.balloons()
                else:
                    st.error(f"❌ **{decision}** (Риск: {prob:.2%})")
                    
                # Вывод SHAP объяснений
                if explanation:
                    st.markdown("### 🔍 Почему модель так решила?")
                    st.write("Топ-5 факторов, повлиявших на решение по этому клиенту:")
                    
                    for item in explanation:
                        feat = item["feature"]
                        impact = item["impact"]
                        
                        # Если impact > 0, значит фича ПОВЫСИЛА риск дефолта
                        if impact > 0:
                            st.warning(f"⬆️ **{feat}** повышает риск (Вклад: +{impact:.2f})")
                        else:
                            st.info(f"⬇️ **{feat}** снижает риск (Вклад: {impact:.2f})")
            else:
                st.error(f"Ошибка сервера: {response.status_code}")
                
        except Exception as e: # ВОТ ЭТОТ БЛОК БЫЛ УТЕРЯН!
            st.error(f"Не удалось подключиться к API: {e}")