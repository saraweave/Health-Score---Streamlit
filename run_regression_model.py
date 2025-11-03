#!/usr/bin/env python3
"""
Train the churn/retain logistic regression model from `Regression Model.ipynb`
against a flat CSV export and generate scored outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:  # scikit-learn >= 1.4
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - fallback for older versions
    FrozenEstimator = None  # type: ignore[assignment]


TAG = "(LOCATION_FEATURE_ACCESS_AND_USAGE_BY_MONTH)"
DEFAULT_DATASET = "0-6 month churn and 12+ month retain since 2024.csv"

DROP_COLUMNS = [
    "Count of FINANCE_ID",
    "ListAggDistinct of Cancel Summary (SWAT_CASE)",
    "Created Date (SWAT_CASE)",
    "filter",
    "Max of Cancel Summary (SWAT_CASE)",
    "SWAT_CASE_ID",
    "SWAT_CASE_STATUS",
    "SWAT_CASE_CREATED_DATE",
    "SWAT_CANCEL_SUMMARY",
    "SWAT_CANCELLATION_REASON",
    "SWAT_CANCELLATION_SUBREASON",
]

COLUMN_ALIASES = {
    "SLUG": "Slug",
    "FINANCE_ID": "Finance Id",
    "LOCATION_ID": "Location Id",
    "LOCATION_NAME": "Location Name",
    "CORE_INDUSTRY": "Core Industry",
    "PRACTICE_MANAGEMENT_SOFTWARE": "Practice Management Software",
    "INTEGRATIONS": "Integrations",
    "STARTING_BUNDLE": "Starting Bundle",
    "END_BUNDLE": "End Bundle",
    "CHURN_OR_RETAIN": "Churn or Retain",
    "CHURN_MONTH": "Churn Month",
    "LIFETIME_MONTHS": "lifetime in Month",
    "STARTING_MRR": "Starting Mrr",
    "LIFETIME_VALUE": "Lifetime Value",
    "FIRST_USAGE_REPORT_MONTH": "First Usage Report Month",
    "FIRST_INBOUND_SMS_COUNT": "Inbound Sms Count",
    "FIRST_OUTBOUND_SMS_COUNT": "Outbound Sms Count",
    "FIRST_AUTOMATED_SMS_COUNT": "Automated Sms Sent Count",
    "FIRST_INBOUND_CALL_COUNT": "Inbound Call Count",
    "FIRST_OUTBOUND_CALL_COUNT": "Outbound Call Count",
    "FIRST_MISSED_TEXT_SMS_COUNT": "Manual Messages Sms Count",
    "REASON_BILLING": "Reason Billing",
    "REASON_PHONE_SYSTEM": "Reason Phone System",
    "REASON_DATA_SYNC": "Reason Data Sync",
}

for week in range(1, 13):
    COLUMN_ALIASES[f"WEEK_{week}_CASES"] = f"Week {week} Cases"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(DEFAULT_DATASET),
        help="Path to the CSV export to model against.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_outputs"),
        help="Directory for scored CSV artifacts.",
    )
    parser.add_argument(
        "--balance-strategy",
        choices=["class_weight", "smote"],
        default="class_weight",
        help="Handle class imbalance via class weights (default) or SMOTE oversampling.",
    )
    return parser.parse_args(argv)


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: COLUMN_ALIASES.get(col, col) for col in df.columns}
    df = df.rename(columns=rename_map)

    # Rename columns that still contain the nested feature tag.
    rename_tag = {}
    for col in df.columns:
        if TAG in col:
            base = col.replace(TAG, "").strip()
            rename_tag[col] = " ".join(base.split())
    if rename_tag:
        df = df.rename(columns=rename_tag)
    return df


def require_columns(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    return [col for col in candidates if col in df.columns]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Restrict to single-location business segment rows.
    segment_col = "BUSINESS_SEGMENT_BY_HIERARCHY_ACTIVE"
    if segment_col in df.columns:
        df = df[
            df[segment_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("single")
        ]

    df = df[df["Churn or Retain"].isin(["Retain", "Churn"])]
    df_retain = df[df["Churn or Retain"] == "Retain"]
    churn_mask = (df["Churn or Retain"] == "Churn") & (
        df["lifetime in Month"].between(1, 6, inclusive="both")
    )
    df_churn = df[churn_mask]
    df_all = pd.concat([df_retain, df_churn], ignore_index=True)

    to_drop = [col for col in DROP_COLUMNS if col in df_all.columns]
    if to_drop:
        df_all = df_all.drop(columns=to_drop)

    df_all = df_all.dropna(
        subset=require_columns(
            df_all, ["Slug", "Finance Id", "Location Id", "Location Name"]
        )
    )
    df_all.dropna(how="all", axis=1, inplace=True)

    week_columns = require_columns(df_all, [f"Week {i} Cases" for i in range(1, 13)])
    if week_columns:
        df_all.dropna(subset=week_columns, how="any", inplace=True)

    usage_columns = require_columns(
        df_all,
        [
            "Inbound Sms Count",
            "Outbound Sms Count",
            "Automated Sms Sent Count",
            "Manual Messages Sms Count",
            "Inbound Call Count",
            "Outbound Call Count",
        ],
    )
    if usage_columns:
        df_all.dropna(subset=usage_columns, how="any", inplace=True)

    for col in usage_columns:
        activated_col = f"{col} Activated"
        df_all[activated_col] = (df_all[col] > 9).astype(int)

    core_bundle_cols = require_columns(
        df_all, ["Core Industry", "Starting Bundle"]
    )
    if core_bundle_cols:
        df_all.dropna(subset=core_bundle_cols, how="any", inplace=True)

    pm_col = df_all.get("Practice Management Software")
    integration_col = df_all.get("Integrations")
    if pm_col is not None or integration_col is not None:
        pm_values = (
            pm_col.astype(str).str.strip()
            if pm_col is not None
            else pd.Series("", index=df_all.index)
        )
        integration_values = (
            integration_col.astype(str).str.strip()
            if integration_col is not None
            else pd.Series("", index=df_all.index)
        )
        missing_both = (pm_values == "") & (integration_values == "")
        df_all = df_all.loc[~missing_both]

    # Bundle categorisation and first-month SWAT volume.
    bundle_conditions = [
        df_all["Starting Bundle"].str.contains("WeaveCore", case=False, na=False),
        df_all["Starting Bundle"].str.contains("WeavePlus", case=False, na=False),
    ]
    bundle_choices = ["core bundle", "plus bundle"]
    df_all["BundleType"] = np.select(bundle_conditions, bundle_choices, default="other")

    first_weeks = require_columns(df_all, [f"Week {i} Cases" for i in range(1, 5)])
    if first_weeks:
        df_all["First Month SWAT Cases"] = df_all[first_weeks].sum(axis=1)
    else:
        df_all["First Month SWAT Cases"] = np.nan

    eod_pattern = (
        r"(Eaglesoft|Open\s*Dental|Dentrix\s*(G5\+?|G6(?:\.2-7)?|G7))"
    )
    integration_series = (
        df_all["Integrations"].astype(str).str.strip()
        if "Integrations" in df_all.columns
        else pd.Series("", index=df_all.index)
    )
    pm_series = (
        df_all["Practice Management Software"].astype(str).str.strip()
        if "Practice Management Software" in df_all.columns
        else pd.Series("", index=df_all.index)
    )
    pms_lookup = pd.Series(
        np.where(
            integration_series != "",
            integration_series,
            pm_series,
        ),
        index=df_all.index,
    )

    df_all["PMS_Group"] = np.where(
        pms_lookup.str.contains(eod_pattern, case=False, na=False),
        "EOD pms",
        "Others",
    )

    reason_cols = require_columns(
        df_all, ["Reason Billing", "Reason Phone System", "Reason Data Sync"]
    )
    for col in reason_cols:
        df_all[f"{col} Flag"] = (df_all[col] > 0).astype(int)

    dummy_cols = require_columns(df_all, ["Core Industry", "PMS_Group", "BundleType"])
    df_model = pd.get_dummies(
        df_all,
        columns=dummy_cols,
        drop_first=False,
        dummy_na=False,
        dtype="int8",
    )

    df_model["churn_flag"] = df_model["Churn or Retain"].map({"Churn": 1, "Retain": 0})
    df_model = df_model.dropna(subset=["churn_flag"])

    return df_all, df_model


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    balance_strategy: str = "class_weight",
) -> Tuple[
    Pipeline,
    CalibratedClassifierCV,
    float,
    dict,
    dict[str, Tuple[pd.DataFrame, pd.Series]],
]:
    X_clean = X.replace([np.inf, -np.inf], np.nan)
    data = pd.concat([y.rename("y"), X_clean], axis=1).dropna(subset=["y"])

    y_all = data["y"].astype(int)
    X_all = data.drop(columns="y")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )

    logit_kwargs = dict(
        penalty="l2",
        solver="saga",
        max_iter=5000,
        random_state=42,
    )
    if "n_jobs" in LogisticRegression().get_params(deep=True):
        logit_kwargs["n_jobs"] = -1
    if balance_strategy == "class_weight":
        logit_kwargs["class_weight"] = "balanced"

    if balance_strategy == "class_weight":
        base_clf = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("logit", LogisticRegression(**logit_kwargs)),
            ]
        )
    else:
        base_clf = ImbPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("smote", SMOTE(random_state=42)),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("logit", LogisticRegression(**logit_kwargs)),
            ]
        )
    base_clf.fit(X_train, y_train)

    calibrator_kwargs = {"method": "isotonic"}
    if FrozenEstimator is not None:
        calibrator_kwargs["estimator"] = FrozenEstimator(base_clf)
    else:  # pragma: no cover - legacy fallback
        calibrator_kwargs["base_estimator"] = base_clf
        calibrator_kwargs["cv"] = "prefit"

    calibrated = CalibratedClassifierCV(**calibrator_kwargs)
    calibrated.fit(X_val, y_val)

    val_proba = calibrated.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, val_proba)

    f1 = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    )
    best_index = np.nanargmax(f1)
    best_thr = float(thresholds[best_index])

    # Evaluate on the held-out test set.
    test_proba = calibrated.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_thr).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_recall_fscore_support(
            y_test, test_pred, average="binary", zero_division=0
        )[0],
        "recall": precision_recall_fscore_support(
            y_test, test_pred, average="binary", zero_division=0
        )[1],
        "f1": precision_recall_fscore_support(
            y_test, test_pred, average="binary", zero_division=0
        )[2],
        "roc_auc": roc_auc_score(y_test, test_proba),
        "pr_auc": average_precision_score(y_test, test_proba),
        "confusion_matrix": confusion_matrix(y_test, test_pred),
        "classification_report": classification_report(
            y_test, test_pred, digits=4
        ),
    }

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    return base_clf, calibrated, best_thr, metrics, splits


def compute_statsmodels_coefficients(
    X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    X_const = sm.add_constant(X_imputed, has_constant="add")

    logit_model = sm.Logit(y, X_const, missing="raise")
    result = logit_model.fit(disp=False)

    summary = result.summary2().tables[1]
    summary = summary.rename(
        columns={
            "Coef.": "coefficient",
            "Std.Err.": "std_err",
            "z": "z_score",
            "P>|z|": "p_value",
        }
    )
    summary.index.name = "variable"
    summary["odds_ratio"] = np.exp(summary["coefficient"])
    summary["pct_odds_change_per_unit"] = (summary["odds_ratio"] - 1.0) * 100.0
    return summary.reset_index()


def score_split(
    X_part: pd.DataFrame,
    y_part: pd.Series,
    split_name: str,
    model: CalibratedClassifierCV,
    thr: float,
    df_reference: pd.DataFrame,
) -> pd.DataFrame:
    proba = model.predict_proba(X_part)[:, 1]
    pred = (proba >= thr).astype(int)

    scored = X_part.copy()
    scored.insert(0, "row_index", X_part.index)

    id_col = None
    for candidate in [
        "account_id",
        "customer_id",
        "location_id",
        "id",
        "AccountId",
        "CustomerId",
        "Location Id",
    ]:
        if candidate in df_reference.columns:
            id_col = candidate
            break

    if id_col:
        scored.insert(1, id_col, df_reference.loc[X_part.index, id_col].values)

    scored.insert(1, "split", split_name)
    scored["y_true"] = y_part.values
    scored["y_proba"] = proba
    scored["y_pred"] = pred
    scored["threshold_used"] = thr

    return scored


def build_risk_table(scored: pd.DataFrame) -> pd.DataFrame:
    proba = scored["y_proba"]

    def col_or_default(name: str) -> pd.Series:
        if name in scored.columns:
            return scored[name].fillna(0)
        return pd.Series(0, index=scored.index)

    core = (col_or_default("BundleType_core bundle") == 1).astype(int)
    eod = (col_or_default("PMS_Group_EOD pms") == 1).astype(int)
    out_sms = col_or_default("Outbound Sms Count Activated").astype(int)
    out_call = col_or_default("Outbound Call Count Activated").astype(int)
    in_sms = col_or_default("Inbound Sms Count Activated").astype(int)
    in_call = col_or_default("Inbound Call Count Activated").astype(int)

    bri = (
        proba
        * (1 + 0.30 * core)
        * (1 + 0.15 * eod)
        * (1 + 0.15 * out_sms)
        * (1 - 0.25 * out_call)
        * (1 - 0.35 * in_sms)
        * (1 - 0.33 * in_call)
    )
    bri = np.clip(bri, 0, 1)

    risk = pd.DataFrame(
        {
            "p_churn": proba,
            "business_risk_index": bri,
        },
        index=scored.index,
    )
    risk["rank_by_p"] = risk["p_churn"].rank(
        ascending=False, method="first"
    ).astype(int)
    risk["rank_by_bri"] = risk["business_risk_index"].rank(
        ascending=False, method="first"
    ).astype(int)
    return risk.sort_values("business_risk_index", ascending=False)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    if not args.csv.exists():
        print(f"Input CSV not found: {args.csv}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.csv)
    df_named = standardise_columns(df_raw)
    df_enriched, df_model = engineer_features(df_named)

    model_features = [
        "Core Industry_Medical",
        "BundleType_core bundle",
        "Week 1 Cases",
        "Week 2 Cases",
        "Week 3 Cases",
        "Week 4 Cases",
        "Inbound Sms Count Activated",
        "Outbound Sms Count Activated",
        "Inbound Call Count Activated",
        "Outbound Call Count Activated",
        "Reason Billing Flag",
        "Reason Data Sync Flag",
        "PMS_Group_EOD pms",
    ]

    available_features = [col for col in model_features if col in df_model.columns]
    if not available_features:
        raise ValueError("No model features found in the dataset after preprocessing.")

    X = df_model[available_features]
    y = df_model["churn_flag"].astype(int)

    base_clf, calibrated, best_thr, metrics, splits = train_model(
        X, y, balance_strategy=args.balance_strategy
    )

    print("=== Test Metrics (threshold tuned on validation) ===")
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Precision (binary): {metrics['precision']:.4f}")
    print(f"Recall (binary)   : {metrics['recall']:.4f}")
    print(f"F1 (binary)       : {metrics['f1']:.4f}")
    print(f"ROC AUC           : {metrics['roc_auc']:.4f}")
    print(f"PR AUC (AvgPrec)  : {metrics['pr_auc']:.4f}")
    print("Confusion Matrix [tn, fp; fn, tp]:")
    print(metrics["confusion_matrix"])
    print("\nClassification report:")
    print(metrics["classification_report"])

    scored_parts = [
        score_split(
            X_part, y_part, split_name, calibrated, best_thr, df_model
        )
        for split_name, (X_part, y_part) in splits.items()
    ]

    common_cols = scored_parts[0].columns
    scored_parts = [df.reindex(columns=common_cols) for df in scored_parts]

    scored_all = pd.concat(scored_parts, axis=0).set_index("row_index")
    scored_meta = scored_all[
        ["split", "y_true", "y_proba", "y_pred", "threshold_used"]
    ].rename(
        columns={
            "y_true": "model_y_true",
            "y_proba": "pred_proba",
            "y_pred": "pred_label",
        }
    )

    df_joined = df_enriched.join(scored_meta, how="left")
    scored_path = args.output_dir / "scored_rows_all_splits.csv"
    df_joined.to_csv(scored_path, index=False)
    print(f"[saved] {scored_path} ({df_joined.shape[0]} rows)")

    scored_dataset = score_split(
        X, y, "all", calibrated, best_thr, df_model
    ).set_index("row_index")
    risk_table = build_risk_table(scored_dataset)
    scored_dataset = scored_dataset.join(risk_table, how="left")

    scored_full = df_enriched.join(
        scored_dataset[
            [
                "split",
                "y_true",
                "y_proba",
                "y_pred",
                "threshold_used",
                "p_churn",
                "business_risk_index",
                "rank_by_p",
                "rank_by_bri",
            ]
        ],
        how="left",
    )

    risk_path = args.output_dir / "scored_dataset_with_risk.csv"
    scored_full.to_csv(risk_path, index=False)
    print(f"[saved] {risk_path} ({scored_full.shape[0]} rows)")

    coef_df = compute_statsmodels_coefficients(X, y)
    coef_path = args.output_dir / "logit_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    print("\n=== Logistic regression coefficients (statsmodels) ===")
    print(coef_df.to_string(index=False))
    print(f"\n[saved] {coef_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
