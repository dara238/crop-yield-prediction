"""
generate_sample_data.py
-----------------------
Creates the SAMPLE (synthetic) dataset used when the app starts.

Five crops: Chilli, Rice, Watermelon, Tomato, Banana.
Each row is one field in one season, with a planting date and a harvest date.

The data is invented so the app can run without real records.
Replace it with real data before drawing real conclusions.

Run from the project folder:
    python data/generate_sample_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
N_ROWS = 2500

# base = typical yield (t/ha); days = typical days from planting to harvest
CROPS = {
    "Rice":       dict(base=3.5,  rain=1300, rain_w=550, temp=28.5, temp_w=4.0, days=115, days_sd=8,
                       fert=110, fert_gain=0.55, irr_gain=0.22, irr_prob=0.40, months=[5, 6, 7], share=0.30),
    "Chilli":     dict(base=9.0,  rain=900,  rain_w=450, temp=26.5, temp_w=4.0, days=130, days_sd=12,
                       fert=140, fert_gain=0.50, irr_gain=0.25, irr_prob=0.55, months=[10, 11, 12], share=0.18),
    "Watermelon": dict(base=24.0, rain=600,  rain_w=400, temp=28.0, temp_w=4.5, days=85,  days_sd=6,
                       fert=120, fert_gain=0.45, irr_gain=0.30, irr_prob=0.60, months=[11, 12, 1], share=0.17),
    "Tomato":     dict(base=28.0, rain=700,  rain_w=400, temp=24.5, temp_w=4.0, days=110, days_sd=10,
                       fert=160, fert_gain=0.50, irr_gain=0.30, irr_prob=0.65, months=[10, 11, 12], share=0.18),
    "Banana":     dict(base=30.0, rain=1800, rain_w=700, temp=27.5, temp_w=4.5, days=330, days_sd=25,
                       fert=200, fert_gain=0.40, irr_gain=0.20, irr_prob=0.35, months=[5, 6, 7], share=0.17),
}
PROVINCES = {"Battambang": 1.06, "Kampong Cham": 1.05, "Takeo": 1.04, "Kandal": 1.03,
             "Siem Reap": 0.97, "Kampot": 0.96, "Pursat": 0.98, "Prey Veng": 1.01}
SOILS = {"Clay": 0.97, "Clay Loam": 1.02, "Loam": 1.07, "Sandy Loam": 1.00, "Sandy": 0.86}


def bell(x, best, width):
    """1.0 at the best value, smaller further away."""
    return np.exp(-0.5 * ((x - best) / width) ** 2)


rows = []
names = list(CROPS)
for crop in RNG.choice(names, size=N_ROWS, p=[CROPS[c]["share"] for c in names]):
    c = CROPS[crop]
    province = RNG.choice(list(PROVINCES))
    soil = RNG.choice(list(SOILS), p=[0.22, 0.22, 0.24, 0.20, 0.12])

    # Planting date: usually in the crop's normal season, sometimes not
    year = int(RNG.integers(2019, 2025))
    month = int(RNG.choice(c["months"])) if RNG.random() < 0.8 else int(RNG.integers(1, 13))
    planting = pd.Timestamp(year=year, month=month, day=int(RNG.integers(1, 29)))
    days = int(np.clip(RNG.normal(c["days"], c["days_sd"]), c["days"] * 0.7, c["days"] * 1.3))
    harvest = planting + pd.Timedelta(days=days)

    rain = float(np.clip(RNG.normal(c["rain"], c["rain"] * 0.28), 150, 3000))
    temp = float(np.clip(RNG.normal(c["temp"] + 0.5, 1.8), 18, 36))
    fert = float(np.clip(RNG.normal(c["fert"], c["fert"] * 0.4), 0, 400))
    irrigated = RNG.random() < c["irr_prob"]
    area = float(np.clip(RNG.lognormal(0.2, 0.7), 0.1, 30))

    eff_rain = rain + (0.5 * (c["rain"] - rain) if irrigated and rain < c["rain"] else 0)
    potential = (
        c["base"]
        * (0.55 + 0.45 * bell(eff_rain, c["rain"], c["rain_w"]))
        * (0.60 + 0.40 * bell(temp, c["temp"], c["temp_w"]))
        * (1 + c["fert_gain"] * (1 - np.exp(-fert / (c["fert"] * 0.9)))) / (1 + c["fert_gain"] * 0.67)
        * (1 + c["irr_gain"] if irrigated else 1.0)
        * PROVINCES[province] * SOILS[soil]
        * (1.0 if month in c["months"] else 0.90)                       # planted in season?
        * (0.88 + 0.12 * bell(days, c["days"] * 1.03, c["days_sd"] * 1.5))  # harvested at the right time?
        * (1 + 0.012 * (year - 2019))                                   # slow improvement over the years
    )
    rows.append({
        "crop_type": crop, "province": province,
        "planting_date": planting.date().isoformat(), "harvest_date": harvest.date().isoformat(),
        "field_area_ha": round(area, 2), "rainfall_mm": round(rain), "avg_temperature_c": round(temp, 1),
        "soil_type": soil, "fertilizer_kg_per_ha": round(fert), "irrigation": "Yes" if irrigated else "No",
        "yield_t_per_ha": round(float(potential * RNG.lognormal(0, 0.09)), 2),
    })

df = pd.DataFrame(rows).astype({"rainfall_mm": float, "fertilizer_kg_per_ha": float})

# A little realistic mess, so the automatic cleaning has something to do
idx = RNG.choice(N_ROWS, 40, replace=False)
df.loc[idx[:20], "crop_type"] = df.loc[idx[:20], "crop_type"].str.lower()
df.loc[idx[20:], "irrigation"] = df.loc[idx[20:], "irrigation"].map({"Yes": "yes", "No": "N"})
df.loc[RNG.choice(N_ROWS, 35, replace=False), "rainfall_mm"] = np.nan
df.loc[RNG.choice(N_ROWS, 20, replace=False), "yield_t_per_ha"] = np.nan
df.loc[RNG.choice(N_ROWS, 3, replace=False), "yield_t_per_ha"] = 9999
df = pd.concat([df, df.sample(20, random_state=1)]).sample(frac=1, random_state=3).reset_index(drop=True)

out = Path(__file__).resolve().parent / "crop_yield_dataset.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
