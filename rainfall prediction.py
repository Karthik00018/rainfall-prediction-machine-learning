

# ============================================================
# CA2 PROJECT
# PREDICTIVE ANALYTICS FOR RAINFALL PREDICTION
# USING MACHINE LEARNING
# ============================================================

# 1. IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# 2. LOAD DATASET

dataset = pd.read_csv(
    r"C:\Users\karth\Downloads\rainfall_area-wt_sd_1901-2015.csv"
)

print("\n========== FIRST FIVE ROWS ==========")
print(dataset.head())

print("\nOriginal dataset shape:", dataset.shape)


# 3. FILTER DATASET

dataset = dataset[
    dataset["YEAR"].between(1948, 2015)
].copy()

dataset = dataset.dropna(subset=["Jun-Sep"])

print("\n========== FILTERED DATASET ==========")
print("Dataset shape:", dataset.shape)
print("Start year:", dataset["YEAR"].min())
print("End year:", dataset["YEAR"].max())


# 4. EXPLORATORY DATA ANALYSIS (EDA)

print("\n========== RAINFALL STATISTICS ==========")
print(dataset["Jun-Sep"].describe())

print("\n========== MISSING VALUES ==========")
print(dataset.isnull().sum())


# GRAPH 1: AVERAGE MONSOON RAINFALL BY YEAR

yearly_rainfall = dataset.groupby("YEAR")["Jun-Sep"].mean()

plt.figure(figsize=(10, 5))
plt.plot(yearly_rainfall.index, yearly_rainfall.values)

plt.title("Average Monsoon Rainfall by Year")
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# GRAPH 2: RAINFALL DISTRIBUTION

plt.figure(figsize=(8, 5))
plt.hist(dataset["Jun-Sep"], bins=20, edgecolor="black")

plt.title("Distribution of Monsoon Rainfall")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# GRAPH 3: AVERAGE RAINFALL BY SUBDIVISION

subdivision_rainfall = (
    dataset.groupby("SUBDIVISION")["Jun-Sep"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(12, 6))
subdivision_rainfall.plot(kind="bar")

plt.title("Average Monsoon Rainfall by Subdivision")
plt.xlabel("Subdivision")
plt.ylabel("Average Rainfall (mm)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# 5. SELECT INPUT FEATURES AND TARGET

# Predict June-September rainfall using:
# Subdivision, Year, and January-May rainfall

features = [
    "SUBDIVISION",
    "YEAR",
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY"
]

X = dataset[features].copy()
y = dataset["Jun-Sep"].copy()


# 6. CHRONOLOGICAL TRAIN-TEST SPLIT

# Training: 1948-2000
# Testing: 2001-2015

train_mask = dataset["YEAR"] <= 2000
test_mask = dataset["YEAR"] > 2000

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

print("\n========== TRAIN-TEST SPLIT ==========")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print(
    "Training years:",
    dataset.loc[train_mask, "YEAR"].min(),
    "to",
    dataset.loc[train_mask, "YEAR"].max()
)

print(
    "Testing years:",
    dataset.loc[test_mask, "YEAR"].min(),
    "to",
    dataset.loc[test_mask, "YEAR"].max()
)


# 7. ENCODE SUBDIVISION AND ALIGN COLUMNS

X_train = pd.get_dummies(
    X_train,
    columns=["SUBDIVISION"],
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=["SUBDIVISION"],
    dtype=int
)

# Ensure training and testing have identical columns

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


# 8. HANDLE MISSING VALUES

# Calculate imputation values using training data only

training_means = X_train.mean()

X_train = X_train.fillna(training_means)
X_test = X_test.fillna(training_means)

print("\nMissing values in training features:",
      X_train.isnull().sum().sum())

print("Missing values in testing features:",
      X_test.isnull().sum().sum())


# 9. CREATE FIVE MACHINE LEARNING MODELS

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=8,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


# 10. EVALUATE MEAN BASELINE MODEL

# Baseline predicts the mean training rainfall
# for every test observation

baseline_prediction = [
    y_train.mean()
] * len(y_test)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_prediction
)

baseline_rmse = mean_squared_error(
    y_test,
    baseline_prediction
) ** 0.5

baseline_r2 = r2_score(
    y_test,
    baseline_prediction
)

print("\n========== MEAN BASELINE ==========")
print("MAE:", round(baseline_mae, 3))
print("RMSE:", round(baseline_rmse, 3))
print("R2 Score:", round(baseline_r2, 3))


# 11. TRAIN AND EVALUATE FIVE MODELS

results_list = []
predictions = {}

for name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Predict test rainfall
    prediction = model.predict(X_test)

    predictions[name] = prediction

    # Calculate evaluation metrics
    mae = mean_absolute_error(
        y_test,
        prediction
    )

    rmse = mean_squared_error(
        y_test,
        prediction
    ) ** 0.5

    r2 = r2_score(
        y_test,
        prediction
    )

    results_list.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("\nModel:", name)
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))


# 12. CREATE MODEL COMPARISON TABLE

results = pd.DataFrame(results_list)

# Sort by lowest RMSE

results = results.sort_values(
    by="RMSE"
).reset_index(drop=True)

print("\n========== MODEL COMPARISON ==========")
print(results.round(3).to_string(index=False))


# 13. COMPARE ALL MODELS WITH BASELINE

baseline_row = pd.DataFrame([{
    "Model": "Mean Baseline",
    "MAE": baseline_mae,
    "RMSE": baseline_rmse,
    "R2 Score": baseline_r2
}])

comparison = pd.concat(
    [results, baseline_row],
    ignore_index=True
)

print("\n========== ALL MODELS AND BASELINE ==========")
print(comparison.round(3).to_string(index=False))


# GRAPH 4: RMSE COMPARISON

plt.figure(figsize=(10, 5))

plt.bar(
    results["Model"],
    results["RMSE"],
    label="Machine Learning Models"
)

plt.axhline(
    y=baseline_rmse,
    linestyle="--",
    label="Mean Baseline"
)

plt.title("RMSE Comparison of Prediction Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("RMSE (mm)")
plt.xticks(rotation=25, ha="right")
plt.legend()
plt.tight_layout()
plt.show()


# GRAPH 5: MAE COMPARISON

plt.figure(figsize=(10, 5))

plt.bar(
    results["Model"],
    results["MAE"]
)

plt.axhline(
    y=baseline_mae,
    linestyle="--",
    label="Mean Baseline"
)

plt.title("MAE Comparison of Prediction Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("MAE (mm)")
plt.xticks(rotation=25, ha="right")
plt.legend()
plt.tight_layout()
plt.show()


# GRAPH 6: R2 SCORE COMPARISON

plt.figure(figsize=(10, 5))

plt.bar(
    results["Model"],
    results["R2 Score"]
)

plt.axhline(
    y=baseline_r2,
    linestyle="--",
    label="Mean Baseline"
)

plt.title("R2 Score Comparison of Prediction Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("R2 Score")
plt.xticks(rotation=25, ha="right")
plt.legend()
plt.tight_layout()
plt.show()


# 14. IDENTIFY THE BEST MODEL

# Best model is selected by the lowest RMSE

best_name = results.iloc[0]["Model"]
best_model = models[best_name]
best_prediction = predictions[best_name]

best_mae = results.iloc[0]["MAE"]
best_rmse = results.iloc[0]["RMSE"]
best_r2 = results.iloc[0]["R2 Score"]

print("\n========== BEST MODEL ==========")
print("Best Model:", best_name)
print("MAE:", round(best_mae, 3))
print("RMSE:", round(best_rmse, 3))
print("R2 Score:", round(best_r2, 3))

if best_rmse < baseline_rmse:
    print("Best model performs better than the baseline.")
else:
    print("Best model does not beat the baseline RMSE.")


# 15. ACTUAL VS PREDICTED RAINFALL

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    best_prediction,
    alpha=0.6
)

# Reference line for perfect predictions

minimum = min(
    y_test.min(),
    best_prediction.min()
)

maximum = max(
    y_test.max(),
    best_prediction.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)

plt.title(
    f"Actual vs Predicted Rainfall - {best_name}"
)

plt.xlabel("Actual Rainfall (mm)")
plt.ylabel("Predicted Rainfall (mm)")
plt.legend()
plt.tight_layout()
plt.show()


# 16. SAVE RESULTS TO CSV FILES

comparison.to_csv(
    "model_results.csv",
    index=False
)

# Save predictions from all five models

prediction_results = pd.DataFrame({
    "Actual_Rainfall": y_test.to_numpy()
})

for name, prediction in predictions.items():
    prediction_results[name] = prediction

prediction_results.to_csv(
    "rainfall_predictions.csv",
    index=False
)

print("\n========== PROJECT COMPLETED ==========")
print("Saved: model_results.csv")
print("Saved: rainfall_predictions.csv")
print("All graphs and model comparisons completed.")
