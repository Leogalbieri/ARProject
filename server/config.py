import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRIMARY_MODEL = "rtdetr-x.pt"
SECONDARY_MODEL = "yolov8x-oiv7.pt"
SEARCH_MODEL = "yolov8x-worldv2.pt"

CONF_THRESHOLD = 0.85
CONF_THRESHOLD_SEARCH = 0.10

SEARCH_TARGET = ["backpack"]

IMG_SIZE = 640

IGNORED_PRIMARY = []

IGNORED_SECONDARY = [
    "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train",
    "Truck", "Boat", "Traffic light", "Fire hydrant", "Stop sign",
    "Parking meter", "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep",
    "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella",
    "Handbag", "Tie", "Suitcase", "Skis", "Snowboard", "Sports ball",
    "Kite", "Baseball bat", "Baseball glove", "Skateboard", "Surfboard",
    "Tennis racket", "Bottle", "Wine glass", "Cup", "Fork", "Knife",
    "Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli",
    "Carrot", "Hot dog", "Pizza", "Cake", "Chair", "Couch", "Bed",
    "Dining table", "Toilet", "Laptop", "Mouse", "Remote control",
    "Keyboard", "Mobile phone", "Microwave oven", "Oven", "Toaster",
    "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors",
    "Teddy bear", "Hair dryer", "Toothbrush", "Boy", "Girl", "Man", "Woman",
    "Human arm", "Human beard", "Human body", "Human ear", "Human eye",
    "Human face", "Human foot", "Human hair", "Human hand", "Human head",
    "Human leg", "Human mouth", "Human nose", "Computer keyboard", "Computer monitor",
    "Computer mouse"
]