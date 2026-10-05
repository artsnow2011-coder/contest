# ==================================================
# AI that predicts solar power generation from weather
# (Render server version for FlutterFlow)
# ==================================================

import os
import numpy as np                  # numerical computing
import pandas as pd                 # working with tables (like Excel)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Fix the random seed so every run gives the same result
tf.keras.utils.set_random_seed(42)


# ==================================================
# Step 1. Read the Excel file
# ==================================================

# solar.xlsx must be in the same folder as main.py (GitHub repository root)
FILE_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solar.xlsx")

raw = pd.read_excel(
    FILE_NAME,
    sheet_name="날짜일시_옆으로통합"   # sheet name inside solar.xlsx
)

# Rename the Korean column headers to English
data = raw.rename(columns={
    "일시": "datetime",
    "인버터1(kWh)": "inverter1",
    "인버터2(kWh)": "inverter2",
    "인버터3(kWh)": "inverter3",
    "인버터4(kWh)": "inverter4",
    "전체합(kWh)": "total_raw",
    "발전시간": "gen_hours",
    "기상_기온(°C)": "temperature",
    "기상_강수량(mm)": "rainfall",
    "기상_풍속(m/s)": "wind_speed",
    "기상_습도(%)": "humidity",
    "기상_현지기압(hPa)": "pressure",
    "기상_일사(MJ/m2)": "solar_radiation",
    "기상_전운량(10분위)": "cloud_cover",
})

# Convert the date text (e.g. "2026-04-01 07") into real dates,
# then sort in time order
data["datetime"] = pd.to_datetime(data["datetime"], format="%Y-%m-%d %H")
data = data.sort_values("datetime").reset_index(drop=True)

# Extract the hour of the day
data["hour"] = data["datetime"].dt.hour


# ==================================================
# Step 2. Clean the data
# ==================================================

# Empty rainfall means no rain, so fill it with 0
data["rainfall"] = data["rainfall"].fillna(0)

# The Excel 'total' column contains some wrong numbers,
# and inverter 1 broke in mid-May and stayed at 0,
# so we add up only inverters 2, 3 and 4, which worked normally.
data["power"] = (
    data["inverter2"]
    + data["inverter3"]
    + data["inverter4"]
)


# ==================================================
# Step 3. Choose inputs and target
# ==================================================

# Information given to the AI = weather + hour
FEATURES = [
    "solar_radiation",   # amount of sunlight (MJ/m2)
    "temperature",       # temperature (°C)
    "humidity",          # humidity (%)
    "wind_speed",        # wind speed (m/s)
    "cloud_cover",       # cloud amount (0 = clear, 10 = overcast)
    "pressure",          # air pressure (hPa)
    "rainfall",          # rainfall (mm)
    "hour",              # hour of the day
]

# What the AI has to predict
TARGET = "power"


# ==================================================
# Step 4. Split into training and test data
# ==================================================

split = int(len(data) * 0.8)

# Make the test period start at 7 a.m. of the next day
# so that a day isn't cut in the middle
while data["hour"][split] != 7:
    split = split + 1

train = data[:split]
test = data[split:]


# ==================================================
# Step 5. Put numbers on the same scale (standardization)
# ==================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(train[FEATURES])
X_test = scaler.transform(test[FEATURES])

y_train = train[TARGET]
y_test = test[TARGET]


# ==================================================
# Step 6. Build the AI (neural network)
# ==================================================

# 8 inputs → 32 neurons → 16 neurons → 1 output (power)

model = Sequential([
    Input(shape=(len(FEATURES),)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1),
])

model.compile(
    optimizer="adam",
    loss="mse"
)


# ==================================================
# Step 7. Train the AI (once, when the server starts)
# ==================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0,
)


# ==================================================
# Step 8~9. Evaluate the AI on data it has never seen
# ==================================================

pred = np.maximum(model.predict(X_test, verbose=0).flatten(), 0)

mae = float(mean_absolute_error(y_test, pred))
r2 = float(r2_score(y_test, pred))


# ==================================================
# Step 12. Correlation between weather and power
# ==================================================

corr = data[FEATURES[:-1]].corrwith(data[TARGET]).sort_values()

RANGES = {name: (float(data[name].min()), float(data[name].max())) for name in FEATURES}


# ==================================================
# Web server (for FlutterFlow)
# ==================================================

app = FastAPI(title="Solar Power Prediction AI")

# Allow FlutterFlow (including web test mode) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Weather(BaseModel):
    solar_radiation: float
    temperature: float
    humidity: float
    wind_speed: float
    cloud_cover: float
    pressure: float
    rainfall: float
    hour: float


@app.get("/")
def home():
    return {
        "message": "Solar power prediction API is running.",
        "epochs_trained": len(history.history["loss"]),
        "mae_kwh": round(mae, 1),
        "r2": round(r2, 2),
    }


@app.get("/correlation")
def correlation():
    return {name: round(float(v), 2) for name, v in corr.items()}


# ==================================================
# Step 13. Predict power for new weather conditions
# ==================================================

@app.post("/predict")
def predict_power(w: Weather):

    values = [
        w.solar_radiation,
        w.temperature,
        w.humidity,
        w.wind_speed,
        w.cloud_cover,
        w.pressure,
        w.rainfall,
        w.hour,
    ]

    # Turn the inputs into a one-row table.
    # The order must match the FEATURES list used for training.
    condition = pd.DataFrame([values], columns=FEATURES)

    # Check whether a value is outside the range the AI learned
    warnings = []
    for name, value in zip(FEATURES, values):
        low, high = RANGES[name]
        if value < low or value > high:
            warnings.append(
                f"{name}={value} is outside training range ({low} ~ {high})"
            )

    # Rescale with the same scaler used in training, then predict
    result = model.predict(scaler.transform(condition), verbose=0)[0][0]

    # Power can't be negative, so return 0 if negative
    result = max(float(result), 0.0)

    return {
        "predicted_power_kwh": round(result, 1),
        "warnings": warnings,
    }
