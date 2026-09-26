# Food Delivery Time Prediction

Predicting food delivery time (in minutes) using order, weather, traffic, and delivery-partner features from a real-world food delivery dataset.

## Dataset

Source: [Food Delivery Dataset](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset) (Kaggle)

- ~45,000 delivery records with delivery partner details, weather, traffic conditions, restaurant/delivery coordinates, and order timestamps.
- Target variable: `Time_taken(min)`

## Data Cleaning & Feature Engineering

- Handled disguised missing values (blank strings, whitespace, placeholder tokens) not caught by pandas' default `isnull()`.
- Standardized inconsistent category labels (e.g. `"conditions Sunny"` → `"Sunny"`) and stripped stray whitespace across categorical columns.
- Engineered **`distance_km`** between restaurant and delivery location using the Haversine formula.
- Engineered **`prep_time_min`** from the gap between order time and pickup time.
- Engineered **`order_hour`** from order timestamp.
- Removed distance outliers (>30 km) identified during EDA.
- Encoded categorical features: one-hot encoding for nominal categories (`Weatherconditions`, `Type_of_order`, `Type_of_vehicle`, `Festival`, `City`), ordinal encoding for `Road_traffic_density` (Low < Medium < High < Jam).

## Exploratory Insights

- **City type** strongly affects delivery time — Semi-Urban deliveries take ~2x longer than Urban.
- **Traffic density**, **weather conditions**, and **multiple deliveries** all show a clear positive relationship with delivery time.
- **Festivals** significantly increase delivery time (~45 min average vs. ~26 min on non-festival days).
- Individual delivery partners show consistent speed differences, suggesting rider-specific skill/experience affects delivery time.

## Modeling

Built a `scikit-learn` pipeline (`ColumnTransformer` + estimator) and compared multiple regression models on a held-out validation split, with hyperparameter tuning via **Optuna**.

| Model | R² | MAE (min) |
|---|---|---|
| Linear / Ridge / Lasso | 0.545 | 5.07 |
| AdaBoost | 0.569 | 4.96 |
| KNN | 0.618 | 4.53 |
| Decision Tree | 0.697 | 3.97 |
| Stacking (RF + GBDT + XGB) | 0.713 | 3.92 |
| Gradient Boosting | 0.720 | 3.91 |
| XGBoost | 0.734 | 3.80 |
| Random Forest | 0.738 | 3.74 |
| **LightGBM (tuned)** | **0.844** | **2.97** |

**Best model: LightGBM**, tuned with Optuna (Bayesian/TPE search over `n_estimators`, `max_depth`, `num_leaves`, `learning_rate`, `subsample`, `colsample_bytree`, and regularization terms).

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (pipelines, encoders, model selection)
- LightGBM, XGBoost
- Optuna (hyperparameter tuning)
- Matplotlib, Seaborn (EDA)

## Possible Next Steps

- Rider-level performance feature (smoothed/target-encoded average delivery time per rider) to capture individual speed differences uncovered during EDA.
- Cyclical encoding of `order_hour` (sin/cos) to better represent time-of-day patterns.
- Further tuning of LightGBM with a larger trial budget.

## Author

Mansi
