"""
recommend.py — recommendations for the farmer.

Idea ("what-if" test): change ONE input, ask the model again, and see
whether the predicted yield goes up. Inputs the farmer can change become
actions; weather inputs become advice on how to plan around them.
"""

import numpy as np
import pandas as pd

from core.cleaning import numeric_columns
from core.config import CONTROLLABLE_WORDS, WEATHER_WORDS


def kind_of(feature: str) -> str:
    low = feature.lower()
    if any(word in low for word in CONTROLLABLE_WORDS):
        return "control"        # the farmer can change it
    if any(word in low for word in WEATHER_WORDS):
        return "weather"        # the farmer can only respond to it
    return "other"              # place, soil type, dates ... : no advice


def label(name: str) -> str:
    return str(name).replace("_", " ").strip().capitalize()


def make_recommendations(model, row: pd.DataFrame, prediction: float, reference: pd.DataFrame, features: list):
    """Returns (recommendations, predicted value if all actions are followed)."""
    recommendations, improved = [], row.copy()
    threshold = max(0.02 * abs(prediction), 1e-9)          # ignore gains under 2%

    for feature in features:
        kind = kind_of(feature)
        if kind == "other" or feature not in reference.columns:
            continue
        current = row.at[0, feature]
        is_number = feature in numeric_columns(reference)

        if is_number:
            values = reference[feature].dropna()
            if values.nunique() < 3:
                continue
            options = np.unique(np.round(np.linspace(values.quantile(0.05), values.quantile(0.95), 15), 1))
        else:
            options = np.array([v for v in reference[feature].dropna().unique() if v != current], dtype=object)
        if len(options) == 0:
            continue

        trial = pd.concat([row] * len(options), ignore_index=True)
        if is_number:      # whole-number columns must accept values such as 90.3
            trial, improved = trial.astype({feature: float}), improved.astype({feature: float})
        trial[feature] = options
        predicted = model.predict(trial)
        best = int(np.argmax(predicted))
        gain = float(predicted[best] - prediction)
        if gain < threshold:
            continue

        best_value, name = options[best], label(feature)
        if is_number:
            verb = "Increase" if best_value > current else "Reduce"
            now, goal = f"{current:,.0f}", f"{best_value:,.0f}"
        else:
            verb, now, goal = "Change", str(current), str(best_value)

        if kind == "control":
            improved.at[0, feature] = best_value
            if "irrig" in feature.lower() and str(best_value) == "Yes":
                text = "Use irrigation if water is available."
            else:
                text = f"{verb} {name.lower()} from {now} to about {goal}."
        else:
            low_side = is_number and current < best_value
            text = (f"{name} ({now}) is {'lower' if low_side else 'higher'} than the level where the model "
                    f"predicts the best result (about {goal}). You cannot change the weather, so plan around it: ")
            if "rain" in feature.lower() or "precip" in feature.lower():
                text += "add irrigation or water harvesting." if low_side else "improve drainage and avoid low-lying plots."
            elif "temp" in feature.lower():
                text += ("shift the planting date to a warmer period." if low_side
                         else "choose a heat-tolerant variety or shift the planting date.")
            else:
                text += "adjust the planting date or variety."
        recommendations.append({"text": text, "gain": gain, "kind": kind})

    recommendations.sort(key=lambda r: r["gain"], reverse=True)
    has_action = any(r["kind"] == "control" for r in recommendations)
    potential = float(model.predict(improved)[0]) if has_action else None
    return recommendations, potential
