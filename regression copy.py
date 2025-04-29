import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

# ----- Shock Simulation Functions -----
def apply_feature_shock(df, shock_level, max_reduction=0.25):
    factor = 1 - (shock_level / 4) * max_reduction
    if "AMT_INCOME_TOTAL" in df.columns:
        df["AMT_INCOME_TOTAL"] *= factor
    if "AMT_ANNUITY" in df.columns:
        df["AMT_ANNUITY"] *= factor
    for ext_col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        if ext_col in df.columns:
            df[ext_col] -= (shock_level / 4) * 0.1
            df[ext_col] = df[ext_col].clip(lower=0)
    return df


def adjust_features_for_new_defaulters(df, flipped_indices, shock_level):
    # Simulate financial stress on the new defaulters
    factor_income_reduction = 0.25  # Reduce income for defaulters
    factor_debt_increase = 0.15  # Increase debt (AMT_ANNUITY) for defaulters
    factor_source_decrease = 0.1  # Reduce EXT_SOURCE values for defaulters

    # Adjust income and annuity for new defaulters
    if "AMT_INCOME_TOTAL" in df.columns:
        df.loc[flipped_indices, "AMT_INCOME_TOTAL"] *= (1 - factor_income_reduction * shock_level)
    if "AMT_ANNUITY" in df.columns:
        df.loc[flipped_indices, "AMT_ANNUITY"] *= (1 + factor_debt_increase * shock_level)

    # Adjust EXT_SOURCE values for new defaulters
    for ext_col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        if ext_col in df.columns:
            df.loc[flipped_indices, ext_col] -= factor_source_decrease * shock_level
            df[ext_col] = df[ext_col].clip(lower=0)  # Avoid negative values for these features

    return df


def inject_label_shock_with_feature_adjustment(df, shock_level, max_flip_rate=0.12):
    flip_rate = (shock_level / 4) * max_flip_rate
    non_defaults = df[df["TARGET"] == 0]
    n_flip = int(flip_rate * len(non_defaults))
    flip_indices = non_defaults.sample(n=n_flip, random_state=shock_level).index
    df.loc[flip_indices, "TARGET"] = 1

    # Adjust features of the newly labeled defaulters
    df = adjust_features_for_new_defaulters(df, flip_indices, shock_level)

    return df

# ----- Initialize results storage -----
summary = []
all_bias_data = []  # ⬅️ Store all bias dataframes

for round_num in range(5):
    print(f"\n=== Processing shock_round_{round_num}.csv ===")

    # Load and apply shocks
    df = pd.read_csv(f"shock_round_{round_num}.csv")
    df = apply_feature_shock(df.copy(), round_num)
    df = inject_label_shock_with_feature_adjustment(df.copy(), round_num)

    # Preprocessing
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) // 365
    df["AGE_GROUP"] = pd.cut(df["AGE_YEARS"], bins=range(20, 72, 3), right=False).astype(str)

    X = df.drop(columns=["SK_ID_CURR", "TARGET"])
    y = df["TARGET"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(
        X, y, df["AGE_GROUP"], test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance
    num_zeros = len(y_train[y_train == 0])
    num_ones = len(y_train[y_train == 1])
    scale_pos_weight = (num_zeros / num_ones) * 0.7 if num_ones > 0 else 1.1

    # Use Logistic Regression instead of XGBoost
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        ))
    ])

    model.fit(X_train, y_train)

    # Predict probabilities and use fixed threshold
    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = 0.65
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    age_group_results = []
    for group in age_test.unique():
        mask = age_test == group
        proportion = y_pred[mask].mean()
        age_group_results.append((group, proportion))

    bias_df = pd.DataFrame(age_group_results, columns=["Age Group", "Positive Rate"]).sort_values("Age Group")
    reference_rate = bias_df["Positive Rate"].max()
    bias_df["DIR"] = bias_df["Positive Rate"] / reference_rate
    bias_df["Shock Round"] = round_num  # ⬅️ Add round number
    all_bias_data.append(bias_df)  # ⬅️ Collect for CSV

    # Save summary
    summary.append({
        "Shock Round": round_num,
        "Accuracy": acc,
        "True Default Rate": y_test.mean(),
        "Predicted Default Rate": y_pred.mean(),
        "Classification Report": report,
        "Confusion Matrix": matrix,
        "Bias DataFrame": bias_df
    })

    # Plot Demographic Parity
    plt.figure(figsize=(10, 5))
    plt.bar(bias_df["Age Group"], bias_df["Positive Rate"], color="skyblue")
    plt.axhline(y=0.8 * reference_rate, color='r', linestyle='--', label='80% Rule Threshold')
    plt.xlabel("Age Group")
    plt.ylabel("Positive Prediction Rate")
    plt.title(f"Demographic Parity - Shock Round {round_num}")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"regression_demographic_parity_round_{round_num}.png")
    plt.close()

    # Save predictions with probabilities
    pred_df = df.loc[X_test.index, ["SK_ID_CURR"]].copy()
    pred_df["Predicted_Prob"] = y_prob
    pred_df["Predicted_TARGET"] = y_pred
    pred_df.to_csv(f"regression_predicted_probs_round_{round_num}.csv", index=False)

# Save summary stats to CSV
summary_df = pd.DataFrame([{
    "Shock Round": s["Shock Round"],
    "Accuracy": s["Accuracy"],
    "True Default Rate": s["True Default Rate"],
    "Predicted Default Rate": s["Predicted Default Rate"]
} for s in summary])

summary_df.to_csv("regression_shock_summary_results.csv", index=False)

# Save combined demographic parity data
combined_bias_df = pd.concat(all_bias_data, ignore_index=True)
combined_bias_df.to_csv("regression_demographic_parity_all_rounds.csv", index=False)

print("\nAll shock round results saved. Check 'regression_shock_summary_results.csv' and demographic parity plots/data.")
