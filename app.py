# ==================================================
# Solar power prediction API (for Render + FlutterFlow)
# ==================================================

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

FILE_NAME = "solar.xlsx"
SHEET_NAME = "날짜일시_옆으로통합"

FEATURES = [
    "solar_radiation",
    "temperature",
    "humidity",
    "wind_speed",
    "cloud_cover",
    "pressure",
    "rainfall",
    "hour",
]
TARGET = "power"


# ---------- Load and clean data ----------
raw = pd.read_excel(FILE_NAME, sheet_name=SHEET_NAME)

data = raw.rename(columns={
    "일시": "datetime",
    "인버터2(kWh)": "inverter2",
    "인버터3(kWh)": "inverter3",
    "인버터4(kWh)": "inverter4",
    "기상_기온(°C)": "temperature",
    "기상_강수량(mm)": "rainfall",
    "기상_풍속(m/s)": "wind_speed",
    "기상_습도(%)": "humidity",
    "기상_현지기압(hPa)": "pressure",
    "기상_일사(MJ/m2)": "solar_radiation",
    "기상_전운량(10분위)": "cloud_cover",
})

data["datetime"] = pd.to_datetime(data["datetime"], format="%Y-%m-%d %H")
data = data.sort_values("datetime").reset_index(drop=True)
data["hour"] = data["datetime"].dt.hour
data["rainfall"] = data["rainfall"].fillna(0)
data[TARGET] = data["inverter2"] + data["inverter3"] + data["inverter4"]


# ---------- Train the model (once, when the server starts) ----------
scaler = StandardScaler()
X = scaler.fit_transform(data[FEATURES])
y = data[TARGET]

model = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=15,
    random_state=42,
)
model.fit(X, y)

RANGES = {name: (float(data[name].min()), float(data[name].max())) for name in FEATURES}


# ---------- Web server ----------
app = Flask(__name__)


# Allow FlutterFlow (including web test mode) to call this API
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def home():
    return jsonify({
        "message": "Solar power prediction API is running.",
        "usage": "POST /predict with JSON body",
        "fields": FEATURES,
        "ranges": RANGES,
    })


@app.route("/predict", methods=["GET", "POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    params = request.get_json(silent=True) if request.method == "POST" else request.args
    params = params or {}

    missing = [name for name in FEATURES if name not in params]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        values = [float(params[name]) for name in FEATURES]
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numbers."}), 400

    warnings = []
    for name, value in zip(FEATURES, values):
        low, high = RANGES[name]
        if value < low or value > high:
            warnings.append(f"{name}={value} is outside training range ({low} ~ {high})")

    condition = pd.DataFrame([values], columns=FEATURES)
    result = model.predict(scaler.transform(condition))[0]
    result = max(float(result), 0.0)

    return jsonify({
        "predicted_power_kwh": round(result, 1),
        "warnings": warnings,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
