import re
import keywords
import config


def process(text):
    print(f"Processing: {text}")

    # Remove caracteres especiais, espaços duplicados e converte para minúsculo
    text = re.sub(r"[^\w\s]", "", " ".join(text.split()).lower())

    print(f"Processed: {text}")

    # Desativa modo reconhecimento
    if any(trigger in text for trigger in keywords.NONE_KEYWORDS):
        config.SELECTED_MODE = "none"
        print("Switching all modes off")

    # Ativa modo reconhecimento
    elif any(trigger in text for trigger in keywords.GENERAL_SEARCH_KEYWORDS):
        config.SELECTED_MODE = "general"
        print("Switching to general mode")

    return config.SELECTED_MODE
