def make_credit_decision(probability_of_default: float) -> str:
    """Бизнес-логика банка: отказываем, если риск дефолта выше 15%."""
    if probability_of_default < 0.15:
        return "Одобрить"
    else:
        return "Отказать"