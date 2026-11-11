"""
models.py — build, train, compare, save and load the models.

Three regression models are compared:
    Random Forest, Gradient Boosting and SVM (Support Vector Regression).
Each model is a pipeline that accepts the raw table, so the same object
is used for training and for prediction.
"""

import io
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from core.features import DateFeatures, build_spec


def make_models(spec: dict, positive_target: bool = True) -> dict:
    """Create the three untrained models.

    When the target is always positive (like yield), the models learn log(target).
    A 10% error then counts the same for a 3 t/ha crop and a 30 t/ha crop, so
    low-yield crops are not ignored.
    """
    def pipeline(regressor):
        prepare = ColumnTransformer([
            ("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), spec["model_numeric"]),
            ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                              ("encode", OneHotEncoder(handle_unknown="ignore"))]), spec["categorical"]),
        ])
        return Pipeline([("dates", DateFeatures(spec["dates"], spec["pair"])),
                         ("prepare", prepare), ("model", regressor)])

    models = {
        "Random Forest": pipeline(RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        "Gradient Boosting": pipeline(GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)),
        # SVR learns best when the target is scaled as well
        "SVM (SVR)": TransformedTargetRegressor(
            regressor=pipeline(SVR(kernel="rbf", C=10, epsilon=0.1)), transformer=StandardScaler()),
    }
    if positive_target:
        models = {name: TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
                  for name, model in models.items()}
    return models


def train_models(df: pd.DataFrame, target: str, features: list, test_size: float = 0.2) -> dict:
    """Train the three models and measure them on rows they did not see."""
    spec = build_spec(df, features)
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=test_size, random_state=42)

    models, rows = make_models(spec, positive_target=bool((df[target] > 0).all())), []
    test = X_test.copy()
    test["__actual__"] = y_test.values

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        test[f"__pred__{name}"] = predicted
        rows.append({"Model": name,
                     "MAE": mean_absolute_error(y_test, predicted),
                     "RMSE": float(np.sqrt(mean_squared_error(y_test, predicted))),
                     "R²": r2_score(y_test, predicted),
                     "Training time (s)": time.time() - start})

    metrics = pd.DataFrame(rows)
    metrics["Accuracy (R² %)"] = metrics["R²"].clip(lower=0) * 100
    best = metrics.sort_values("RMSE").iloc[0]["Model"]              # lowest error wins

    # Which inputs matter most? Shuffle one column at a time and see how much the score drops.
    try:
        sample = X_test.sample(min(len(X_test), 400), random_state=1)
        result = permutation_importance(models[best], sample, y_test.loc[sample.index],
                                        n_repeats=3, random_state=42, scoring="r2")
        importance = pd.Series(result.importances_mean, index=features).clip(lower=0).sort_values(ascending=False)
    except Exception:
        importance = None

    return {"models": models, "metrics": metrics, "best": best, "target": target, "features": features,
            "spec": spec, "test": test.reset_index(drop=True), "importance": importance,
            "n_train": len(X_train), "n_test": len(X_test)}


def bundle_bytes(results: dict, name: str) -> bytes:
    """Pack one trained model so it can be downloaded and uploaded again later."""
    row = results["metrics"].set_index("Model").loc[name]
    bundle = {"app": "crop_yield_app", "model_name": name, "model": results["models"][name],
              "features": results["features"], "target": results["target"],
              "metrics": {"MAE": float(row["MAE"]), "RMSE": float(row["RMSE"]), "R2": float(row["R²"])}}
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer, compress=3)
    return buffer.getvalue()


def read_model_file(file) -> dict:
    """Load an uploaded model file. Raises ValueError with a clear message."""
    try:
        loaded = joblib.load(file)
    except Exception as error:
        raise ValueError(f"This is not a valid model file ({type(error).__name__}).") from error

    if isinstance(loaded, dict) and "model" in loaded and "features" in loaded:
        model, features, target = loaded["model"], list(loaded["features"]), loaded.get("target")
    else:
        model, target = loaded, None
        features = [str(f) for f in getattr(loaded, "feature_names_in_", [])]

    if not hasattr(model, "predict"):
        raise ValueError("The file does not contain a model with a predict() method.")
    if not features:
        raise ValueError("The model does not say which input columns it needs. Upload a model saved "
                         "by this app, or a scikit-learn model that was trained on a DataFrame.")
    return {"name": file.name, "model": model, "features": features, "target": target}
