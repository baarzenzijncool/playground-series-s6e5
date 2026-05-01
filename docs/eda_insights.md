# EDA Insights: Playground Series S6E5 — F1 Pit Stop Prediction

**Target**: `PitNextLap` — binary (1 = driver will pit on or near the next lap).  
**Dataset**: 439,140 train rows | 188,165 test rows | 26 unique circuits | 887 drivers | 4 years (2022–2025).

---

## Theme 1: Target Variable & Class Imbalance

### Finding 1.1 — Moderate Imbalance
> **Observation**: 19.9% of laps have PitNextLap=1; imbalance ratio is 4:1 (no-pit : pit).  
> **Hypothesis**: Pit events are relatively frequent in this dataset compared to real F1 (where pitting is ~1–3 times per race, or 1–3 laps out of 60). The 19.9% base rate suggests PitNextLap=1 is flagged for multiple consecutive laps approaching a pit stop window, not just the single lap immediately before the pit.  
> **Evidence**: 18.7% of driver-stints contain more than one PitNextLap=1 lap; max observed is 30 per stint. A purely "next-lap" label would give ≤5% base rate.  
> **Modeling decision**: Use ROC-AUC and F1 as primary metrics. Set `class_weight='balanced'` or `scale_pos_weight=4` for gradient boosting. Avoid accuracy — a constant-zero classifier scores 80%.

### Finding 1.2 — 2023 Year Anomaly (Critical)
> **Observation**: 2023 has a pit rate of **0.96%** vs 26–30% for 2022, 2024, and 2025. 2023 contributes 136,147 rows (31% of train) but only ~1,307 pit events.  
> **Hypothesis**: 2023 data was generated with different labeling logic, or "Pre-Season Testing" events (included in 2023: 7,855 rows with near-zero pit rates) have contaminated the year-level signal. This creates a severe distribution shift within the training set.  
> **Risk**: A model trained without accounting for this will learn to predict near-zero pit probability for any row with Year=2023. If test set has 2023 rows with real pit patterns, predictions will be wrong.  
> **Action**: (a) Treat Year=2023 as a categorical with special handling; (b) investigate/exclude Pre-Season Testing rows; (c) if using `Year` as a feature, do NOT use ordinal encoding — use one-hot or target encoding with heavy smoothing.

---

## Theme 2: Tyre Life — The Primary Pit Signal

### Finding 2.1 — Compound-Specific TyreLife Cliffs
> **Observation**: TyreLife is the single strongest predictor (Pearson r=0.274). Pit probability increases monotonically with TyreLife and then spikes sharply past compound-specific thresholds:

| Compound | Median TyreLife at Pit | 75th pct (cliff) | Pit Rate |
|----------|----------------------|------------------|----------|
| SOFT | 12 laps | 16 laps | 19.3% |
| MEDIUM | 16 laps | 22 laps | 10.1% |
| HARD | 20 laps | 27 laps | 32.8% |
| INTERMEDIATE | 17 laps | 24 laps | 15.2% |
| WET | 11 laps | 17 laps | 2.5% |

> **Hypothesis**: Each compound degrades at a different rate; teams pit before or at the compound-specific cliff where lap-time loss exceeds track-position loss. The cliff is not a fixed threshold but a distribution with team strategy determining exactly when in the distribution each team acts.  
> **Suggested features**:
> - `tyre_life_ratio = TyreLife / compound_median_pit_life` (normalizes stint age across compounds; SOFT Q50=12, MEDIUM=16, HARD=20)
> - `tyre_life_over_cliff` (binary: TyreLife > compound 75th pct at pit — i.e., 16/22/27 for S/M/H)
> - `tyre_life_remaining_to_cliff = compound_q75_pit_life − TyreLife` (negative = overdue for pit)

### Finding 2.2 — HARD Compound Usage Pattern (Surprising)
> **Observation**: HARD has the highest per-lap pit rate (32.8%) despite being the most durable compound.  
> **Hypothesis**: HARD is used as the primary "workhorse" compound in long first/second stints. Because these stints are the longest in absolute laps, the "pit window" (multiple consecutive laps with PitNextLap=1) generates more pit-labeled rows proportionally. HARD also appears in short stints after safety cars and early strategy pits, creating short stints with early PitNextLap=1.  
> **Modeling decision**: Do not treat compound hardness (SOFT < MEDIUM < HARD) as a simple linear encoding. The relationship between compound and pit probability is non-monotonic — use one-hot encoding or a target-encoded feature.

---

## Theme 3: Cumulative Degradation

### Finding 3.1 — Degradation Is a Nonlinear Threshold Signal
> **Observation**: Cumulative_Degradation has r=−0.167 with PitNextLap. More negative values (higher degradation) correlate with higher pit probability. The relationship is nonlinear — there is a threshold beyond which pit probability spikes.  
> **Hypothesis**: Teams accept gradual degradation but react quickly once a critical level is crossed (exponential performance loss on the "cliff"). The per-lap degradation rate (deg/lap) is a better signal than raw cumulative degradation.  
> **Evidence**: Mean deg_rate for pit laps = −2.76 vs −3.08 for no-pit laps (small but consistent difference). Note: high outlier deg_rate values (e.g., max 1191 for pit vs 2410 for no-pit) suggest noisy telemetry rows.  
> **Suggested features**:
> - `degradation_rate = Cumulative_Degradation / max(TyreLife, 1)` (average deg per lap)
> - `degradation_above_compound_q90` (binary: compound-normalized, flags extreme degradation events)
> - `degradation_vs_compound_median` = Cumulative_Degradation − compound_median_cumulative_deg (signed distance from expected)

---

## Theme 4: Lap Time Delta

### Finding 4.1 — LapTime_Delta Separates Classes at the Median
> **Observation**: Lap time delta at pit laps (PitNextLap=1) has median=−4.3s vs −0.14s for no-pit laps. Pearson correlation with PitNextLap is near-zero (−0.005), but the distributional shift at the median is meaningful.  
> **Hypothesis**: LapTime_Delta is right-skewed with heavy outliers (max 2,396–2,423s) that mask the median signal. The relevant signal is whether the driver is running slower than the reference lap (delta > 0), not the magnitude. Large positive deltas (slower pace) indicate tyre wear; the extreme negative outliers (safety car laps) add noise.  
> **Suggested features**:
> - `laptime_delta_pos` (binary: LapTime_Delta > 0, i.e., slower than theoretical reference)
> - `laptime_delta_clipped = clip(LapTime_Delta, −20, 30)` (remove safety-car outliers)
> - `laptime_delta_above_compound_q75` (compound-normalized threshold: is this lap unusually slow for this compound?)
> - `laptime_trend` = rolling 3-lap mean of LapTime_Delta, if lap-by-lap join is possible (captures acceleration of degradation rather than instantaneous reading)

---

## Theme 5: Race Progress & Strategic Pit Windows

### Finding 5.1 — Two Distinct Pit Windows
> **Observation**: Pits cluster in two windows. Distribution of RaceProgress at pit laps: Q1=0.264, median=0.437, Q3=0.590. Pit rate after RaceProgress>0.85 collapses to **7.5%** (vs 19.9% overall). A small spike exists for RaceProgress<0.05 (lap-1 incident pits, rate=5.3%).  
> **Hypothesis**: Primary window (26–45% race progress) corresponds to the first strategic stop (fresh tyres for the second stint). Secondary window (55–70%) corresponds to a second stop or early-fuel laps where fresh rubber gives maximum pace. After 85% of the race, pitting loses all strategic value.  
> **Suggested features**:
> - `in_primary_pit_window` (binary: RaceProgress ∈ [0.25, 0.45])
> - `in_secondary_pit_window` (binary: RaceProgress ∈ [0.52, 0.72])
> - `too_late_to_pit` (binary: RaceProgress > 0.85)
> - `race_phase` (categorical: OPENING [0–0.05] / EARLY [0.05–0.35] / MID [0.35–0.65] / LATE [0.65–0.85] / CLOSING [>0.85])

---

## Theme 6: Race Position & Undercut Dynamics

### Finding 6.1 — Mid-Field Drivers Pit More
> **Observation**: Average race position at pit laps = 9.9 (IQR: P6–P14). Pit rate varies by position, with mid-field and backmarker positions pitting more frequently than the front.  
> **Hypothesis**: Race leaders protect track position by staying out; mid-field drivers use pit stops as a strategic weapon (undercut). Drivers losing places (negative Position_Change) have additional incentive to pit and attempt an overcut or undercut reset.  
> **Suggested features**:
> - `position_group` (categorical: P1-3, P4-6, P7-10, P11-15, P16+)
> - `losing_positions` (binary: Position_Change < −1)
> - `undercut_pressure = Position × RaceProgress × (1 − RaceProgress)` (peaks mid-race for mid-field)

---

## Theme 7: PitStop Flag — Just-Pitted Signal

### Finding 7.1 — PitStop Flag Is a Near-Perfect Negative Indicator at Short Timescales
> **Observation**: When PitStop=1 (pitted this lap), PitNextLap=1 rate = **24.8%** vs 19.1% when PitStop=0. Pearson r=+0.049.  
> **Hypothesis**: Counterintuitive: the current PitStop=1 rows showing elevated PitNextLap=1 suggests these are not standard "pitted this lap" rows but rather rows where the pit window spans the current and next lap (consistent with PitNextLap being a multi-lap window label). `PitStop` is still informative but not the near-zero-probability signal one would expect if "pitting this lap" meant "definitely NOT pitting next lap."  
> **Suggested features**:
> - `just_pitted = PitStop` (direct feature — small but real signal)
> - `laps_since_last_pit` (reconstruct from Stint transitions and TyreLife; captures "freshness" of tyres)

---

## Theme 8: Stint Number & Strategy Structure

### Finding 8.1 — Stint 2 Is the Prime Pit Trigger
> **Observation**: Pit rate by stint:

| Stint | Pit Rate | Row Count |
|-------|----------|-----------|
| 1 | 5.98% | 216,288 |
| 2 | 39.1% | 129,536 |
| 3 | 29.3% | 69,238 |
| 4 | 17.2% | 18,903 |
| 5+ | <6% | <5,000 |

> **Hypothesis**: Stint 1 is usually the longest (often starting on HARD for strategy), with pits concentrated at its end. Stint 2 has the highest per-lap pit rate because the transition from Stint 2 → Stint 3 is the most common single pit stop in a 2-stop race. Stints 5+ rarely lead to another pit.  
> **Suggested features**:
> - `stint_encoded` (target encoding or ordinal: Stint 2–4 have non-monotonic pit rates, so ordinal is wrong)
> - `is_likely_final_stint` (binary: Stint ≥ 4 OR estimated_race_laps × (1−RaceProgress) < compound_expected_stint)
> - `stints_completed = Stint − 1` (number of pit stops already made)

---

## Theme 9: Driver & Team Patterns

### Finding 9.1 — Massive Driver-Level Variance
> **Observation**: Driver pit rates range from 0% to 56.6% (std=9.85%) across 887 unique drivers. This spread is larger than any single numeric feature.  
> **Hypothesis**: Driver codes are proxies for constructor teams; team strategy philosophy (aggressive undercuts vs. long-stint endurance) dominates individual driver behavior. Some team codes appear with 0% pit rate (likely testing/reserve drivers with few laps) while race-regulars cluster between 15–35%.  
> **Suggested features**:
> - `driver_pit_rate_encoded` (target encoding: mean PitNextLap per driver, mandatory 5-fold CV to prevent leakage)
> - `driver_avg_tyre_life_at_pit` (team stint-length preference, CV-encoded)
> - Apply smoothing: `encoded = (count × mean + global_mean × k) / (count + k)`, k~20.

---

## Theme 10: Circuit & Year Effects

### Finding 10.1 — 4× Difference in Pit Rate Across Circuits
> **Observation**: Pit rate ranges from **9.1% (Mexico City)** to **38.9% (Chinese GP)** across 26 circuits (std=7.8%). High-deg: Chinese GP (38.9%), Monaco (35.7%), Spanish GP (32.0%), Bahrain (28.8%). Low-deg: Mexico City (9.1%), Miami (10.4%), Austin (11.4%), Monza (13.2%).

> **Hypothesis**: Circuit characteristics (altitude, abrasive tarmac, corner loading) determine tyre degradation rate. High-altitude Mexico City reduces tyre thermal loading → long stints. Street circuits like Monaco surprisingly have high pit rates, possibly due to safety-car driven pit windows.

> **Suggested features**:
> - `race_pit_rate_encoded` (CV target encoding; fallback to year-level mean for any unseen race in test — confirmed no unseen races in test for this competition)
> - `race_laps_total` (max LapNumber per race; affects what "late race" means)
> - `year_target_encoded` — NOT ordinal, due to the 2023 anomaly (0.96% vs 26–30% for other years).

---

## Theme 11: Correlation & Feature Ranking

### Summary — Linear Correlations with PitNextLap

| Feature | Pearson r |
|---------|-----------|
| TyreLife | +0.274 |
| LapNumber | +0.267 |
| Stint | +0.198 |
| RaceProgress | +0.186 |
| Cumulative_Degradation | −0.167 |
| Year | +0.125 |
| PitStop | +0.049 |
| Position_Change | +0.046 |
| LapTime (s) | −0.034 |
| Position | +0.021 |
| LapTime_Delta | −0.005 |

> **Note**: TyreLife, LapNumber, and RaceProgress are all positively correlated with PitNextLap and also correlated with each other (all track "how far into the race are we"). Mutual information (capturing non-linear relationships) is expected to rank TyreLife even higher.

### Finding 11.1 — LapTime_Delta Linear Correlation Is Misleading
> **Observation**: LapTime_Delta has near-zero Pearson correlation (−0.005) but the **median** at pit laps (−4.3s) differs substantially from no-pit laps (−0.14s).  
> **Hypothesis**: The correlation is masked by extreme outlier laps (safety car in-laps have deltas of −2,400s) which swamp the Pearson calculation. The median-based signal is real and should be captured via a clipped or rank-transformed version of this feature.

---

## Theme 12: Key Interaction Effects

### Finding 12.1 — TyreLife × Compound Is the Core Interaction
> **Observation**: A SOFT tyre at TyreLife=15 has very high pit probability; a HARD tyre at TyreLife=15 has low probability. The interaction is the dominant feature of the problem.  
> **Suggested features**:
> - `tyre_life_ratio` (= TyreLife / compound median) already captures this
> - `compound × tyre_life_bin` (explicit interaction for linear models)

### Finding 12.2 — Position × RaceProgress (Undercut Window)
> **Observation**: Pit rates are highest at P7-P14 during the 30–70% race progress window — the undercut zone. Outside that window, mid-field pit rates converge with the baseline.  
> **Suggested features**:
> - `undercut_zone = (Position > 6) & (RaceProgress ∈ [0.30, 0.70])` (binary interaction flag)

### Finding 12.3 — TyreLife × Degradation (Urgency Signal)
> **Observation**: High TyreLife AND high absolute Cumulative_Degradation jointly predict pit probability better than either alone. The heatmap shows cells with both conditions having 2–3× the pit rate vs either condition alone.  
> **Suggested features**:
> - `tyre_stress = tyre_life_ratio × abs(degradation_rate)` (compound-normalized combined urgency)

---

## Theme 13: Train / Test Distribution

### Finding 13.1 — No Unseen Circuits in Test
> **Observation**: All races in the test set appear in the training set. Years 2022–2025 appear in both. No domain shift on categorical features.  
> **Implication**: Race and driver target encodings can be computed on training data and applied directly to test without an unseen-category fallback. However, the **2023 year effect** (0.96% pit rate) must be handled carefully: year-level target encoding will suppress pit predictions for all 2023 test rows.

### Finding 13.2 — Feature Distributions Are Stable Across Train/Test
> **Observation**: Visual inspection shows TyreLife, RaceProgress, Position, LapTime_Delta, and LapNumber have similar distributions in train and test. No obvious covariate shift.  
> **Implication**: Standard CV (k-fold) is appropriate; no need for time-based or adversarial validation for numeric features. The 2023 effect is the main distribution concern.

---

## Feature Engineering Roadmap

Features listed by priority (estimated impact on model performance).

### Priority 1 — Critical (build first)

| Feature | Formula / Method | Rationale |
|---------|-----------------|-----------|
| `tyre_life_ratio` | `TyreLife / compound_median_pit_life` | Normalizes TyreLife across compounds; single strongest feature |
| `tyre_life_over_cliff` | `TyreLife > compound_q75_pit_life` (binary) | Flags laps past the "must pit now" threshold |
| `compound_encoded` | One-hot (5 categories) | Compound is non-ordinal; HARD behaves differently from expected |
| `year_target_encoded` | CV target encoding with smoothing | 2023 anomaly makes ordinal/one-hot risky |
| `race_phase` | Categorical bucketing of RaceProgress | Captures pit window structure better than raw RaceProgress |
| `is_likely_final_stint` | `Stint >= 4` (binary) | Stints 4+ have <18% pit rate; strong negative predictor |

### Priority 2 — High Impact

| Feature | Formula / Method | Rationale |
|---------|-----------------|-----------|
| `degradation_rate` | `Cumulative_Degradation / max(TyreLife, 1)` | Per-lap deg rate; better than raw cumulative |
| `driver_pit_rate_encoded` | CV target encoding (5-fold, k=20 smoothing) | Captures team strategy; 0–57% spread |
| `race_pit_rate_encoded` | CV target encoding (5-fold, k=10 smoothing) | 4× difference across circuits |
| `laptime_delta_clipped` | `clip(LapTime_Delta, −20, 30)` | Removes safety-car outliers that corrupt linear models |
| `in_pit_window` | `RaceProgress ∈ [0.25,0.45] or [0.52,0.72]` | Direct window flag |
| `too_late_to_pit` | `RaceProgress > 0.85` | Hard cutoff; pit rate drops to 7.5% |
| `stint_target_encoded` | CV target encoding | Stint 2 = 39% vs Stint 1 = 6%; non-ordinal |

### Priority 3 — Medium Impact

| Feature | Formula / Method | Rationale |
|---------|-----------------|-----------|
| `losing_positions` | `Position_Change < −1` (binary) | Falling-back drivers pit more |
| `position_group` | Categorical bucket (P1-3, P4-6, P7-10, P11+) | Mid-field pits most |
| `tyre_life_remaining_to_cliff` | `compound_q75_pit_life − TyreLife` | Signed distance to pit threshold |
| `tyre_stress` | `tyre_life_ratio × abs(degradation_rate)` | Interaction for linear models |
| `just_pitted` | `= PitStop` | Slight positive signal (24.8% vs 19.1%) |
| `laps_since_last_pit` | Reconstruct from TyreLife/Stint transitions | "Freshness" proxy |

### Priority 4 — Low / Refinement

| Feature | Formula / Method | Rationale |
|---------|-----------------|-----------|
| `closing_lap_flag` | `RaceProgress > 0.90` (binary) | Refines `too_late_to_pit` |
| `laptime_delta_above_compound_q75` | Per-compound normalized threshold | Captures "slow for this compound" |
| `undercut_zone` | `(Position > 6) & (RaceProgress ∈ [0.30,0.70])` | Mid-race mid-field interaction |
| `driver_avg_tyre_life_at_pit` | Mean TyreLife at pit per driver (CV-encoded) | Team stint-length preference |
| `race_laps_total` | Max LapNumber per (Race, Year) | Needed for any "laps remaining" calc |
| `year_2023_flag` | `Year == 2023` (binary) | Explicitly marks anomalous year for tree models |

---

*Generated from `notebooks/eda.ipynb`. Investigate 2023 pit-rate anomaly (0.96%) before training.*
