from src.utils import make_credit_decision

def test_credit_approval():
    # Тест 1: Идеальный клиент (риск 5%)
    assert make_credit_decision(0.05) == "Одобрить"

def test_credit_rejection():
    # Тест 2: Рискованный клиент (риск 20%)
    assert make_credit_decision(0.20) == "Отказать"

def test_borderline_case():
    # Тест 3: Клиент прямо на границе (ровно 15%)
    assert make_credit_decision(0.15) == "Отказать"