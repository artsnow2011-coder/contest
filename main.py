# ==================================================
# AI that predicts solar power generation from weather
# ==================================================

import os
import numpy as np                  # numerical computing
import pandas as pd                 # working with tables (like Excel)
import matplotlib.pyplot as plt     # drawing graphs

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Fix the random seed so every run gives the same result
tf.keras.utils.set_random_seed(42)


# ==================================================
# Upload the file (Only if solar.xlsx does not exist)
# ==================================================
if not os.path.exists("solar.xlsx"):
    from google.colab import files
    print("solar.xlsx 파일이 존재하지 않아 업로드 창을 띄웁니다.")
    uploaded = files.upload()
    FILE_NAME = list(uploaded.keys())[0] if uploaded else "solar.xlsx"
else:
    print("이미 solar.xlsx 파일이 업로드되어 있어 기존 파일을 사용합니다.")
    FILE_NAME = "solar.xlsx"


# ==================================================
# Step 1. Read the Excel file
# ==================================================

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

print("Total number of rows:", len(data))


# ==================================================
# Step 4. Split into training and test data
# ==================================================

# The earlier 80% of the data is used for training,
# and the later 20% is kept aside for the final test.
#
# Why we don't shuffle the time order:
# in real life we can't know future weather and power in advance,
# so we learn from the past and predict what comes after.

split = int(len(data) * 0.8)

# Make the test period start at 7 a.m. of the next day
# so that a day isn't cut in the middle
while data["hour"][split] != 7:
    split = split + 1

train = data[:split]
test = data[split:]

print("Training rows:", len(train), "/ Test rows:", len(test))
print(
    "Test period:",
    test["datetime"].min().date(),
    "~",
    test["datetime"].max().date()
)


# ==================================================
# Step 5. Put numbers on the same scale (standardization)
# ==================================================

# Pressure is around 1000 while rainfall is around 0~10,
# so the sizes of the numbers are very different.
# We rescale every input so the AI isn't biased by number size.
#
# The scale is learned from the training data only.

scaler = StandardScaler()

X_train = scaler.fit_transform(train[FEATURES])
X_test = scaler.transform(test[FEATURES])

y_train = train[TARGET]
y_test = test[TARGET]


# ==================================================
# Step 6. Build the AI (neural network)
# ==================================================

# 8 inputs
#     ↓
# 32 neurons
#     ↓
# 16 neurons
#     ↓
# 1 output (power)

model = Sequential([
    Input(shape=(len(FEATURES),)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1),
])

# adam: method that adjusts weights to reduce the error
# mse : method that measures the gap between prediction and actual value
model.compile(
    optimizer="adam",
    loss="mse"
)


# ==================================================
# Step 7. Train the AI
# ==================================================

# 20% of the training data is not used for learning directly
# but as 'validation data'.
#
# While training, we check whether the validation error also goes down,
# to see if the AI is just memorizing the training data.
#
# If the validation error doesn't improve 15 times in a row,
# training stops, so we don't train longer than needed.

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=300,               # train up to 300 rounds
    batch_size=32,            # learn 32 rows at a time
    validation_split=0.2,     # use 20% of training data for validation
    callbacks=[early_stop],
    verbose=0,
)

print(
    "Epochs actually trained:",
    len(history.history["loss"])
)


# ==================================================
# Step 8. Predict power on data the AI has never seen
# ==================================================

pred = model.predict(
    X_test,
    verbose=0
).flatten()

# Power can't be negative, so negative predictions become 0
pred = np.maximum(pred, 0)


# Put the results in a table
results = pd.DataFrame({
    "datetime": test["datetime"].values,
    "actual": y_test.values,
    "predicted": pred.round(1),
})

# Difference between prediction and actual value
results["error"] = (
    results["predicted"] - results["actual"]
).round(1)

print()
print(results.head(12))

# Save the results as a CSV file
results.to_csv(
    "ai_power_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# Step 9. Evaluate the AI
# ==================================================

# Mean Absolute Error (MAE):
# how far, on average, the AI's prediction is from the actual power.
# Smaller is better.
#
# Coefficient of determination (R²):
# how well the AI explains the changes in actual power.
# Closer to 1 is better.

mae = mean_absolute_error(
    y_test,
    pred
)

r2 = r2_score(
    y_test,
    pred
)

print()
print("====== AI Performance ======")
print(f"Mean Absolute Error : {mae:.1f} kWh")
print(f"R²                  : {r2:.2f}")
print(
    f"(Reference) Mean power in test period : "
    f"{y_test.mean():.1f} kWh"
)


# ==================================================
# Step 10. Graph 1: actual vs predicted power
# ==================================================

# Show the first 5 days of the test period
# 12 hours a day × 5 days = 60 points

view = results.head(60)

plt.figure(figsize=(12, 5))

plt.plot(
    view["actual"].values,
    "o-",
    label="Actual power"
)

plt.plot(
    view["predicted"].values,
    "s--",
    label="AI predicted power"
)

plt.xlabel("Time order (one day = 7:00~18:00, 12 points)")
plt.ylabel("Power (kWh)")
plt.title("Actual vs AI Predicted Power (first 5 days of test period)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "graph1_actual_vs_predicted.png",
    dpi=200
)

plt.show()


# ==================================================
# Step 11. Graph 2: actual vs predicted scatter
# ==================================================

# The closer the points are to the diagonal line,
# the closer the AI's predictions are to the actual values.

plt.figure(figsize=(6, 6))

plt.scatter(
    results["actual"],
    results["predicted"],
    s=12,
    alpha=0.6
)

max_val = max(
    results["actual"].max(),
    results["predicted"].max()
)

plt.plot(
    [0, max_val],
    [0, max_val],
    "r--",
    label="Actual = Predicted"
)

plt.xlabel("Actual power (kWh)")
plt.ylabel("AI predicted power (kWh)")

plt.title(
    f"Actual vs AI Predicted Power (R² = {r2:.2f})"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "graph2_scatter.png",
    dpi=200
)

plt.show()


# ==================================================
# Step 12. Which weather factors matter most?
# ==================================================

# Analyze only weather factors (exclude hour)
weather = FEATURES[:-1]

# Correlation between each weather factor and power
corr = (
    data[weather]
    .corrwith(data[TARGET])
    .sort_values()
)

print()
print("====== Weather vs Power (correlation) ======")
print(corr.round(2))


# Show correlations as a bar chart
plt.figure(figsize=(9, 5))

colors = [
    "tomato" if v < 0 else "steelblue"
    for v in corr.values
]

plt.barh(
    corr.index,
    corr.values,
    color=colors
)

plt.axvline(
    0,
    color="black",
    linewidth=0.8
)

plt.xlabel(
    "Correlation "
    "(positive = rise together, "
    "negative = move in opposite directions)"
)

plt.title(
    "Weather Factors vs Solar Power"
)

plt.tight_layout()

plt.savefig(
    "graph3_correlation.png",
    dpi=200
)

plt.show()


print()
print("Analysis complete!")


# ==================================================
# Step 13. Predict power for new weather conditions
# ==================================================

# Function: enter weather conditions,
# and the AI calculates the expected power

def predict_power(
    solar_radiation,
    temperature,
    humidity,
    wind_speed,
    cloud_cover,
    pressure,
    rainfall,
    hour
):

    # Turn the inputs into a one-row table.
    # The order must match the FEATURES list used for training.

    condition = pd.DataFrame(
        [[
            solar_radiation,
            temperature,
            humidity,
            wind_speed,
            cloud_cover,
            pressure,
            rainfall,
            hour
        ]],
        columns=FEATURES
    )

    # Check whether a value is outside the range the AI learned
    for name in FEATURES:

        value = condition[name][0]

        low = data[name].min()
        high = data[name].max()

        if value < low or value > high:

            print(
                f"  ⚠ {name} = {value} "
                f"is outside the training range "
                f"({low} ~ {high}), "
                f"so the prediction may be inaccurate."
            )

    # Rescale with the same scaler used in training
    condition_scaled = scaler.transform(condition)

    # Let the AI predict the power
    value = model.predict(
        condition_scaled,
        verbose=0
    )[0][0]

    # Power can't be negative, so return 0 if negative
    return max(
        float(value),
        0
    )


# ==================================================
# Step 14. Try several weather conditions
# ==================================================

print()
print("====== Predictions by Condition ======")


# Sunny day
sunny = predict_power(
    solar_radiation=3.0,
    temperature=28,
    humidity=40,
    wind_speed=2.5,
    cloud_cover=0,
    pressure=1003,
    rainfall=0,
    hour=12
)

print(
    f"Sunny day, 12:00  : "
    f"{sunny:.1f} kWh"
)


# Cloudy day
cloudy = predict_power(
    solar_radiation=0.8,
    temperature=24,
    humidity=80,
    wind_speed=2.0,
    cloud_cover=10,
    pressure=998,
    rainfall=0,
    hour=12
)

print(
    f"Cloudy day, 12:00 : "
    f"{cloudy:.1f} kWh"
)


# Rainy day
rainy = predict_power(
    solar_radiation=0.3,
    temperature=22,
    humidity=95,
    wind_speed=3.0,
    cloud_cover=10,
    pressure=995,
    rainfall=5,
    hour=12
)

print( 
    f"Rainy day, 12:00  : "
    f"{rainy:.1f} kWh"
)


# ==================================================
# Step 15. Enter your own weather conditions
# ==================================================

# Type in each weather condition to get a prediction.
#
# Type q in the solar radiation box to stop.

print()
print("====== Predict Power from Your Own Input ======")

print(
    "(Reference ranges)"
)

print(
    "Solar radiation 0~4, "
    "Temperature 3~38, "
    "Humidity 11~99, "
    "Wind speed 0~6, "
    "Cloud cover 0~10, "
    "Pressure 986~1013, "
    "Rainfall 0~14, "
    "Hour 7~18"
)

# Non-interactive fallback logic for non-interactive runner
try:
    solar_radiation = input("\nSolar radiation (MJ/m2) [q to quit or press Enter to skip] : ")
    if solar_radiation and solar_radiation.lower() != "q":
        temperature = input("Temperature (°C)        : ")
        humidity = input("Humidity (%)            : ")
        wind_speed = input("Wind speed (m/s)        : ")
        cloud_cover = input("Cloud cover (0~10)      : ")
        pressure = input("Pressure (hPa)          : ")
        rainfall = input("Rainfall (mm)           : ")
        hour = input("Hour (7~18)             : ")

        result = predict_power(
            float(solar_radiation),
            float(temperature),
            float(humidity),
            float(wind_speed),
            float(cloud_cover),
            float(pressure),
            float(rainfall),
            float(hour)
        )
        print(f"👉 AI predicted power : {result:.1f} kWh")
except (KeyboardInterrupt, EOFError):
    print("\nInteractive input skipped.")
