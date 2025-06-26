import requests
from datetime import datetime
import pytz
import pandas as pd
import os
from dotenv import load_dotenv

# 📍 Chemin absolu de ton projet
BASE_DIR = "/Users/fredericmendessemedo/Desktop/Projet_TomTom"

# 📦 Chemins complets des fichiers
LOG_PATH = os.path.join(BASE_DIR, "cron_log.txt")
CSV_PATH = os.path.join(BASE_DIR, "traffic_data.csv")
ENV_PATH = os.path.join(BASE_DIR, ".env")

# 🔐 Charger la clé API depuis le fichier .env
load_dotenv(ENV_PATH)
API_KEY = os.getenv('API_KEY')

# 📍 Coordonnées GPS
LAT, LON = 48.8566, 2.3522  # Paris
RADIUS = 10

# 🕒 Date/heure locale Paris
now = datetime.now(pytz.timezone("Europe/Paris"))

def get_traffic_row():
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{RADIUS}/json?point={LAT},{LON}&unit=KMPH&key={API_KEY}"
    r = requests.get(url)
    if r.status_code != 200:
        with open(LOG_PATH, "a") as f:
            f.write(f"[{now.isoformat()}] ERREUR API: {r.status_code}\n")
        return None

    flow = r.json()["flowSegmentData"]
    speed = flow["currentSpeed"]
    free = flow["freeFlowSpeed"]
    confidence = flow["confidence"]

    ratio = speed / free if free > 0 else 0
    congestion = (
        "LOW" if ratio > 0.85 else
        "MEDIUM" if ratio > 0.5 else
        "HIGH"
    )

    return {
        "timestamp": now.isoformat(),
        "hour": now.hour,
        "day_of_week": now.weekday(),
        "date": now.date().isoformat(),
        "currentSpeed": speed,
        "freeFlowSpeed": free,
        "confidence": confidence,
        "congestion": congestion
    }

# ✅ Log de démarrage du script
with open(LOG_PATH, "a") as f:
    f.write(f"[{now.isoformat()}] Script exécuté - API_KEY présente : {API_KEY is not None}\n")

if __name__ == "__main__":
    row = get_traffic_row()
    if row:
        df = pd.DataFrame([row])
        file_exists = os.path.exists(CSV_PATH)
        df.to_csv(CSV_PATH, mode="a", index=False, header=not file_exists)
