import re
import keywords
import mode_state


def process(text):
    print(f"Processing: {text}")

    # Clean text
    text = re.sub(r"[^\w\s]", "", " ".join(text.split()).lower())

    print(f"Processed: {text}")

    # Disable all modes
    if any(trigger in text for trigger in keywords.NONE_KEYWORDS):
        mode_state.set_mode("none")
        print("Switching all modes off")

    # Enable general recognition mode
    elif any(trigger in text for trigger in keywords.GENERAL_SEARCH_KEYWORDS):
        mode_state.set_mode("general")
        print("Switching to general mode")
