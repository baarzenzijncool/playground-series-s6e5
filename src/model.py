import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_COLS, build_features, clean_columns, fit_encodings

DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "scale_pos_weight": 4,  # 4:1 class imbalance (EDA Finding 1.1)
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 30,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "device": "cpu",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    train = clean_columns(train)
    test = clean_columns(test)
    return train, test


def run_cv(train: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """5-fold CV. Returns OOF predictions and best iteration per fold."""
    X = train.drop(columns=["id", "pitnextlap"])
    y = train["pitnextlap"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train))
    best_iters: list[int] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        encodings = fit_encodings(X_tr, y_tr)
        X_tr_feat = build_features(X_tr, encodings)[FEATURE_COLS]
        X_val_feat = build_features(X_val, encodings)[FEATURE_COLS]

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_tr_feat, y_tr,
            eval_set=[(X_val_feat, y_val)],
            verbose=200,
        )

        oof_preds[val_idx] = model.predict_proba(X_val_feat)[:, 1]
        best_iters.append(model.best_iteration)
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"Fold {fold + 1} | AUC: {fold_auc:.5f} | best_iter: {model.best_iteration}")

    fold_aucs = [
        roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])
        for _, val_idx in KFold(n_splits=5, shuffle=True, random_state=42).split(X)
    ]
    oof_auc = roc_auc_score(y, oof_preds)
    std = np.std(fold_aucs)
    print(f"\nOOF AUC: {oof_auc:.5f} ± {std:.5f}")
    return oof_preds, best_iters


def train_full_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    best_iters: list[int],
) -> np.ndarray:
    X = train.drop(columns=["id", "pitnextlap"])
    y = train["pitnextlap"]
    X_test = test.drop(columns=["id"])

    avg_iters = int(np.mean(best_iters))
    print(f"\nTraining on full data with n_estimators={avg_iters}")

    encodings = fit_encodings(X, y)
    X_feat = build_features(X, encodings)[FEATURE_COLS]
    X_test_feat = build_features(X_test, encodings)[FEATURE_COLS]

    params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    params["n_estimators"] = avg_iters

    model = xgb.XGBClassifier(**params)
    model.fit(X_feat, y, verbose=200)

    return model.predict_proba(X_test_feat)[:, 1]


def save_submission(test: pd.DataFrame, preds: np.ndarray, oof_auc: float) -> Path:
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fname = SUBMISSIONS_DIR / f"submission_xgb_auc{oof_auc:.4f}_{timestamp}.csv"
    pd.DataFrame({"id": test["id"], "PitNextLap": preds}).to_csv(fname, index=False)
    print(f"Saved: {fname}")
    return fname


def main() -> None:
    train, test = load_data()
    oof_preds, best_iters = run_cv(train)
    oof_auc = roc_auc_score(train["pitnextlap"], oof_preds)
    test_preds = train_full_and_predict(train, test, best_iters)
    save_submission(test, test_preds, oof_auc)


if __name__ == "__main__":
    main()
