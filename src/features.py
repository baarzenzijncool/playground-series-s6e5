import numpy as np
import pandas as pd

# Compound tyre life thresholds from EDA (Finding 2.1)
COMPOUND_MEDIAN_PIT_LIFE = {
    "SOFT": 12, "MEDIUM": 16, "HARD": 20, "INTERMEDIATE": 17, "WET": 11,
}
COMPOUND_Q75_PIT_LIFE = {
    "SOFT": 16, "MEDIUM": 22, "HARD": 27, "INTERMEDIATE": 24, "WET": 17,
}


def _add_bigrams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bi_compound_race"]   = df["compound"] + "__" + df["race"]
    df["bi_compound_stint"]  = df["compound"] + "__" + df["stint"].astype(str)
    df["bi_driver_compound"] = df["driver"]   + "__" + df["compound"]
    df["bi_race_year"]       = df["race"]     + "__" + df["year"].astype(str)
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that require no target-level information from training data."""
    df = df.copy()
    df = _add_bigrams(df)
    rp = df["raceprogress"]
    compound = df["compound"]

    # Priority 1 — tyre life
    median_life = compound.map(COMPOUND_MEDIAN_PIT_LIFE).fillna(15).clip(lower=1)
    q75_life = compound.map(COMPOUND_Q75_PIT_LIFE).fillna(20)
    df["tyre_life_ratio"] = df["tyrelife"] / median_life
    df["tyre_life_over_cliff"] = (df["tyrelife"] > q75_life).astype(int)
    df["tyre_life_remaining_to_cliff"] = q75_life - df["tyrelife"]

    # Priority 1 — race phase / pit windows
    df["race_phase"] = np.select(
        [rp < 0.05, rp < 0.35, rp < 0.65, rp < 0.85],
        [0, 1, 2, 3],
        default=4,
    )
    df["in_pit_window"] = (
        ((rp >= 0.25) & (rp <= 0.45)) | ((rp >= 0.52) & (rp <= 0.72))
    ).astype(int)
    df["too_late_to_pit"] = (rp > 0.85).astype(int)
    df["closing_lap_flag"] = (rp > 0.90).astype(int)

    # Priority 1 — stint
    df["is_likely_final_stint"] = (df["stint"] >= 4).astype(int)
    df["stints_completed"] = df["stint"] - 1

    # Priority 2 — degradation
    df["degradation_rate"] = df["cumulative_degradation"] / df["tyrelife"].clip(lower=1)
    df["tyre_stress"] = df["tyre_life_ratio"] * df["degradation_rate"].abs()

    # Priority 2 — lap time
    df["laptime_delta_clipped"] = df["laptime_delta"].clip(-20, 30)

    # Priority 3 — position
    df["losing_positions"] = (df["position_change"] < -1).astype(int)
    df["position_group"] = pd.cut(
        df["position"].fillna(20),
        bins=[0, 3, 6, 10, 15, 100],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)
    df["undercut_zone"] = (
        (df["position"] > 6) & (rp >= 0.30) & (rp <= 0.70)
    ).astype(int)

    # Priority 3 — pit recency
    df["just_pitted"] = df["pitstop"]

    # Priority 4 — year anomaly flag
    df["year_2023_flag"] = (df["year"] == 2023).astype(int)

    # Priority 1 — compound one-hot (non-ordinal per EDA Finding 2.2)
    for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        df[f"compound_{c.lower()}"] = (compound == c).astype(int)

    return df


def fit_encodings(df: pd.DataFrame, y: pd.Series) -> dict:
    """Learn target encodings and lookup tables from a training split."""
    global_mean = float(y.mean())

    def target_enc(series, k):
        stats = y.groupby(series).agg(["sum", "count"])
        return ((stats["sum"] + global_mean * k) / (stats["count"] + k)).to_dict()

    df_bi = _add_bigrams(df)

    # Per-compound laptime_delta q75 (computed on clipped values to match test-time feature)
    laptime_delta_clipped = df["laptime_delta"].clip(-20, 30)
    laptime_q75 = laptime_delta_clipped.groupby(df["compound"]).quantile(0.75).to_dict()

    # Max lap number per race / year — used for race_laps_total feature
    race_laps = (
        df.groupby(["race", "year"])["lapnumber"]
        .max()
        .reset_index()
        .rename(columns={"lapnumber": "race_laps_total"})
    )

    return {
        "global_mean": global_mean,
        "driver_enc":             target_enc(df["driver"],                k=20),
        "race_enc":               target_enc(df["race"],                  k=10),
        "year_enc":               target_enc(df["year"],                  k=15),
        "stint_enc":              target_enc(df["stint"],                 k=10),
        "bi_compound_race_enc":   target_enc(df_bi["bi_compound_race"],   k=10),
        "bi_compound_stint_enc":  target_enc(df_bi["bi_compound_stint"],  k=10),
        "bi_driver_compound_enc": target_enc(df_bi["bi_driver_compound"], k=15),
        "bi_race_year_enc":       target_enc(df_bi["bi_race_year"],       k=5),
        "laptime_q75_by_compound": laptime_q75,
        "race_laps": race_laps,
    }


def apply_encodings(df: pd.DataFrame, encodings: dict) -> pd.DataFrame:
    """Apply precomputed encodings — must be called after add_static_features."""
    df = df.copy()
    gm = encodings["global_mean"]

    df["driver_encoded"] = df["driver"].map(encodings["driver_enc"]).fillna(gm)
    df["race_encoded"]   = df["race"].map(encodings["race_enc"]).fillna(gm)
    df["year_encoded"]   = df["year"].map(encodings["year_enc"]).fillna(gm)
    df["stint_encoded"]  = df["stint"].map(encodings["stint_enc"]).fillna(gm)

    df["bi_compound_race_encoded"]   = df["bi_compound_race"].map(encodings["bi_compound_race_enc"]).fillna(gm)
    df["bi_compound_stint_encoded"]  = df["bi_compound_stint"].map(encodings["bi_compound_stint_enc"]).fillna(gm)
    df["bi_driver_compound_encoded"] = df["bi_driver_compound"].map(encodings["bi_driver_compound_enc"]).fillna(gm)
    df["bi_race_year_encoded"]       = df["bi_race_year"].map(encodings["bi_race_year_enc"]).fillna(gm)

    laptime_q75 = df["compound"].map(encodings["laptime_q75_by_compound"]).fillna(0)
    df["laptime_delta_above_compound_q75"] = (
        df["laptime_delta_clipped"] > laptime_q75
    ).astype(int)

    fallback_laps = int(df["lapnumber"].max())
    df = df.merge(encodings["race_laps"], on=["race", "year"], how="left")
    df["race_laps_total"] = df["race_laps_total"].fillna(fallback_laps)

    return df


def build_features(df: pd.DataFrame, encodings: dict) -> pd.DataFrame:
    df = add_static_features(df)
    df = apply_encodings(df, encodings)
    return df


FEATURE_COLS = [
    # Raw numeric
    "tyrelife", "lapnumber", "raceprogress", "cumulative_degradation",
    "position", "position_change", "laptime_s", "year",
    # Tyre life
    "tyre_life_ratio", "tyre_life_over_cliff", "tyre_life_remaining_to_cliff", "tyre_stress",
    # Race progress
    "race_phase", "in_pit_window", "too_late_to_pit", "closing_lap_flag",
    # Stint
    "stint", "is_likely_final_stint", "stints_completed",
    # Degradation
    "degradation_rate",
    # Lap time
    "laptime_delta_clipped", "laptime_delta_above_compound_q75",
    # Position
    "position_group", "losing_positions", "undercut_zone",
    # Pit recency
    "just_pitted",
    # Year
    "year_2023_flag",
    # Compound one-hot
    "compound_soft", "compound_medium", "compound_hard", "compound_intermediate", "compound_wet",
    # Target encoded — base
    "driver_encoded", "race_encoded", "year_encoded", "stint_encoded",
    # Target encoded — bigram cross-features
    "bi_compound_race_encoded", "bi_compound_stint_encoded",
    "bi_driver_compound_encoded", "bi_race_year_encoded",
    # Race context
    "race_laps_total",
]
