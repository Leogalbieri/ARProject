import json
import os

STATE_MODE_FILE = "state.json"


def get_mode():
    if not os.path.exists(STATE_MODE_FILE):
        return "none"
    with open(STATE_MODE_FILE, "r") as f:
        return json.load(f).get("mode", "none")


def set_mode(mode):
    with open(STATE_MODE_FILE, "w") as f:
        json.dump({"mode": mode}, f)
