"""
config.py — settings and reference knowledge used by the whole app.
Change values here; no other file needs editing.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PATH = BASE_DIR / "data" / "crop_yield_dataset.csv"
ASSETS_DIR = BASE_DIR / "assets"

APP_TITLE = "Crop Yield Prediction"
SAMPLE_NAME = "Sample dataset (synthetic)"
PAGES = ["🏠 Home", "🧠 Train Model", "📈 Prediction", "📤 Upload & Train", "ℹ️ About"]

MIN_ROWS = 30            # smallest dataset accepted for training
MAX_CATEGORIES = 50      # text columns with more values are not used as inputs
UPLOAD_TYPES = ["csv", "tsv", "txt", "xlsx", "xls", "json", "ipynb"]
MODEL_TYPES = ["pkl", "joblib"]

GREEN = "#2d6a4f"
ACCENT = "#e09f3e"
PALETTE = ["#2d6a4f", "#e09f3e", "#3a7ca5", "#9e2a2b", "#6d597a", "#588157", "#bc6c25", "#457b9d"]

YES_NO = {"yes": "Yes", "y": "Yes", "true": "Yes", "no": "No", "n": "No", "false": "No"}

# Words in a column name that tell the app what kind of column it is
TARGET_WORDS = ("yield", "production", "output", "harvest_kg", "target")
CONTROLLABLE_WORDS = ("fertil", "nitrogen", "phosph", "potass", "npk", "irrig", "manure",
                      "lime", "pestic", "seed_rate", "compost")
WEATHER_WORDS = ("rain", "precip", "temp", "humid", "sun", "solar")
START_DATE_WORDS = ("plant", "sow", "seed", "transplant", "start")
END_DATE_WORDS = ("harvest", "end", "pick")

# Typical growth stages: (stage name, days). Approximate; they vary by variety and season.
GROWTH_STAGES = {
    "Rice": [("Vegetative", 55), ("Reproductive", 35), ("Ripening", 30)],
    "Chilli": [("Seedling", 30), ("Vegetative", 35), ("Flowering", 25), ("Fruiting and harvest", 40)],
    "Watermelon": [("Seedling", 15), ("Vine growth", 25), ("Flowering", 15), ("Fruit development", 30)],
    "Tomato": [("Seedling", 25), ("Vegetative", 25), ("Flowering", 20), ("Fruiting and harvest", 40)],
    "Banana": [("Establishment", 60), ("Vegetative", 150), ("Flowering", 30), ("Bunch development", 90)],
}
CROP_ALIASES = {
    "rice": "Rice", "paddy": "Rice", "paddy rice": "Rice",
    "chilli": "Chilli", "chili": "Chilli", "chillies": "Chilli", "chilies": "Chilli", "chile": "Chilli",
    "watermelon": "Watermelon", "watermelons": "Watermelon",
    "tomato": "Tomato", "tomatoes": "Tomato",
    "banana": "Banana", "bananas": "Banana",
}
CROP_TIPS = {
    "Rice": "Keep the field flooded or moist from panicle initiation to flowering. "
            "Water stress at flowering causes empty grains.",
    "Chilli": "Keep the soil evenly moist during flowering and avoid waterlogging. "
              "Water stress makes the flowers drop.",
    "Watermelon": "Give steady water during flowering and fruit set, then reduce watering "
                  "as the fruits ripen to improve sweetness.",
    "Tomato": "Stake the plants and water at the base to keep the leaves dry. "
              "Very hot nights reduce fruit set.",
    "Banana": "Bananas need steady water all year. Remove extra suckers so each plant "
              "keeps one or two followers, and protect the plants from strong wind.",
}


def known_crop(name):
    """Return the standard crop name ('tomatoes' -> 'Tomato'), or None if unknown."""
    return CROP_ALIASES.get(str(name).strip().lower())
