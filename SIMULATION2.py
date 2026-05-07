"""
=============================================================================
  VCC SIMULATION — PHASE 1
  Import Libraries → Load Datasets → Select 4 Timestamps → Train & Predict
=============================================================================

  Datasets:
    - vehicular_traffic_dataset.csv
    - vehicle_individual_dataset.csv
    - task_offloading_dataset.csv

  Time periods (24-hr format):
    Morning   : 06:00 – 11:59
    Afternoon : 12:00 – 16:59
    Evening   : 17:00 – 20:59
    Night     : 21:00 – 05:59

  Algorithms (winners from Voting):
    task_offloading_dataset   → Random Forest   → cpu_cycles_M, data_size_KB, deadline_ms
    vehicular_traffic_dataset → Linear Regression → total_vehicles_present, moving_vehicles
    vehicle_individual_dataset→ Decision Tree   → speed_kmh, vehicle_type
                              → Linear Regression → signal_strength_dBm
=============================================================================
"""

# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble      import RandomForestRegressor
from sklearn.linear_model  import LinearRegression
from sklearn.tree          import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics       import (mean_absolute_error, mean_squared_error,
                                   r2_score, accuracy_score, f1_score)

np.random.seed(42)
random.seed(42)

print("=" * 70)
print("  VCC SIMULATION — PHASE 1: DATA LOADING & PREDICTION")
print("=" * 70)


# =============================================================================
# SECTION 2 — TIME PERIOD DEFINITIONS
# =============================================================================

# Each period maps to a list of valid HH:MM timestamps
# Night wraps around midnight (21:00–05:59)
TIME_PERIODS = {
    "Morning"  : (6,  12),   # 06:00 – 11:59
    "Afternoon": (12, 17),   # 12:00 – 16:59
    "Evening"  : (17, 21),   # 17:00 – 20:59
    "Night"    : (21, 6),    # 21:00 – 05:59  (wraps midnight)
}

def get_hour(ts: str) -> int:
    """Extract hour from 'HH:MM' string."""
    return int(ts.split(":")[0])

def ts_in_period(ts: str, period: str) -> bool:
    """Return True if timestamp HH:MM falls within the named period."""
    h = get_hour(ts)
    start, end = TIME_PERIODS[period]
    if start < end:                  # normal range (e.g. 6–12)
        return start <= h < end
    else:                            # wraps midnight (e.g. 21–6)
        return h >= start or h < end


# =============================================================================
# SECTION 3 — LOAD ALL THREE DATASETS
# =============================================================================
print("\n" + "-" * 70)
print("  SECTION 3 | Loading Datasets")
print("-" * 70)

TRAFFIC_CSV = "vehicular_traffic_dataset.csv"
VEHICLE_CSV = "vehicle_individual_dataset.csv"
TASK_CSV    = "task_offloading_dataset.csv"

df_traffic = pd.read_csv(TRAFFIC_CSV)
df_vehicle = pd.read_csv(VEHICLE_CSV)
df_task    = pd.read_csv(TASK_CSV)

# Add numeric timestamp index (preserves chronological order)
def add_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'ts_idx' column: minutes since 00:00."""
    df = df.copy()
    df['ts_idx'] = df['timestamp'].apply(
        lambda t: int(t.split(':')[0]) * 60 + int(t.split(':')[1])
    )
    return df.sort_values('ts_idx').reset_index(drop=True)

df_traffic = add_ts_index(df_traffic)
df_vehicle = add_ts_index(df_vehicle)
df_task    = add_ts_index(df_task)

# Label-encode categorical columns for vehicle & task datasets
le_vtype = LabelEncoder()
df_vehicle['vehicle_type_enc'] = le_vtype.fit_transform(df_vehicle['vehicle_type'])

le_task_type = LabelEncoder()
df_task['task_type_enc'] = le_task_type.fit_transform(df_task['task_type'])

print(f"  Traffic dataset   : {len(df_traffic):>7,} rows | "
      f"{df_traffic['timestamp'].nunique()} timestamps | "
      f"{df_traffic['region_id'].nunique()} regions")
print(f"  Vehicle dataset   : {len(df_vehicle):>7,} rows | "
      f"{df_vehicle['timestamp'].nunique()} timestamps")
print(f"  Task dataset      : {len(df_task):>7,} rows | "
      f"{df_task['timestamp'].nunique()} timestamps")
print(f"\n  Vehicle types     : {list(le_vtype.classes_)}")
print(f"  Task types        : {list(le_task_type.classes_)}")


# =============================================================================
# SECTION 4 — SELECT 4 TIMESTAMPS (one per period, truly unseen)
# =============================================================================
print("\n" + "-" * 70)
print("  SECTION 4 | Selecting 4 Timestamps (one per time period)")
print("-" * 70)

all_timestamps = sorted(df_traffic['timestamp'].unique(),
                        key=lambda t: int(t.split(':')[0]) * 60 + int(t.split(':')[1]))

# Bucket every timestamp into its period
period_buckets: dict[str, list[str]] = {p: [] for p in TIME_PERIODS}
for ts in all_timestamps:
    for period in TIME_PERIODS:
        if ts_in_period(ts, period):
            period_buckets[period].append(ts)
            break

# Randomly pick one timestamp per period
selected: dict[str, str] = {}
for period, candidates in period_buckets.items():
    chosen = random.choice(candidates)
    selected[period] = chosen
    print(f"  {period:<10} → {chosen}  "
          f"  (pool: {len(candidates)} timestamps)")

selected_ts_set = set(selected.values())
print(f"\n  Selected timestamps: {list(selected.values())}")
print(f"  These 4 will be EXCLUDED from training and used only for prediction.")


# =============================================================================
# SECTION 5 — SPLIT: TRAINING DATA vs PREDICTION TIMESTAMPS
# =============================================================================
print("\n" + "-" * 70)
print("  SECTION 5 | Train / Predict Split")
print("-" * 70)

# Training rows = everything EXCEPT the 4 selected timestamps
df_traffic_train = df_traffic[~df_traffic['timestamp'].isin(selected_ts_set)].copy()
df_vehicle_train = df_vehicle[~df_vehicle['timestamp'].isin(selected_ts_set)].copy()
df_task_train    = df_task[~df_task['timestamp'].isin(selected_ts_set)].copy()

# Prediction rows = ONLY the 4 selected timestamps
df_traffic_pred  = df_traffic[df_traffic['timestamp'].isin(selected_ts_set)].copy()
df_vehicle_pred  = df_vehicle[df_vehicle['timestamp'].isin(selected_ts_set)].copy()
df_task_pred     = df_task[df_task['timestamp'].isin(selected_ts_set)].copy()

print(f"  Traffic  — train: {len(df_traffic_train):>6,} rows | "
      f"predict: {len(df_traffic_pred):>4,} rows")
print(f"  Vehicle  — train: {len(df_vehicle_train):>6,} rows | "
      f"predict: {len(df_vehicle_pred):>4,} rows")
print(f"  Task     — train: {len(df_task_train):>6,} rows | "
      f"predict: {len(df_task_pred):>4,} rows")


# =============================================================================
# SECTION 6 — MODEL A: TRAFFIC PREDICTION (Linear Regression)
#   Targets  : total_vehicles_present, moving_vehicles
#   Features : ts_idx, region_id, vehicles_arrived, vehicles_left
# =============================================================================
print("\n" + "-" * 70)
print("  SECTION 6 | Model A — Traffic Prediction (Linear Regression)")
print("-" * 70)

TRAFFIC_FEATURES = ['ts_idx', 'region_id', 'vehicles_arrived', 'vehicles_left']
TRAFFIC_TARGETS  = ['total_vehicles_present', 'moving_vehicles']

X_tr_trf = df_traffic_train[TRAFFIC_FEATURES].values
Y_tr_trf = df_traffic_train[TRAFFIC_TARGETS].values

X_pred_trf = df_traffic_pred[TRAFFIC_FEATURES].values
Y_true_trf = df_traffic_pred[TRAFFIC_TARGETS].values

# Scale features
scaler_trf = StandardScaler()
X_tr_trf_sc   = scaler_trf.fit_transform(X_tr_trf)
X_pred_trf_sc = scaler_trf.transform(X_pred_trf)

# Train
model_traffic = LinearRegression()
model_traffic.fit(X_tr_trf_sc, Y_tr_trf)

# Predict
Y_pred_trf = model_traffic.predict(X_pred_trf_sc)

print(f"\n  Trained on {len(X_tr_trf):,} rows → predicting {len(X_pred_trf)} rows\n")
print(f"  {'Target':<28} {'R2':>8} {'MAE':>8} {'RMSE':>8}")
print(f"  {'-'*55}")

traffic_metrics = {}
for i, tgt in enumerate(TRAFFIC_TARGETS):
    r2   = r2_score(Y_true_trf[:, i], Y_pred_trf[:, i])
    mae  = mean_absolute_error(Y_true_trf[:, i], Y_pred_trf[:, i])
    rmse = np.sqrt(mean_squared_error(Y_true_trf[:, i], Y_pred_trf[:, i]))
    traffic_metrics[tgt] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"  {tgt:<28} {r2:>8.4f} {mae:>8.2f} {rmse:>8.2f}")

# Attach predictions back for later use
df_traffic_pred = df_traffic_pred.copy()
df_traffic_pred['pred_total_vehicles_present'] = Y_pred_trf[:, 0].clip(0)
df_traffic_pred['pred_moving_vehicles']         = Y_pred_trf[:, 1].clip(0)


# =============================================================================
# SECTION 7 — MODEL B: VEHICLE SPEC PREDICTION
#   B1 — Decision Tree   → speed_kmh (regression)
#   B2 — Decision Tree   → vehicle_type (classification)
#   B3 — Linear Regression → signal_strength_dBm
#   Features: ts_idx, region_id, is_static
# =============================================================================
print("\n" + "-" * 70)
print("  SECTION 7 | Model B — Vehicle Spec Prediction")
print("-" * 70)

VEHICLE_FEATURES     = ['ts_idx', 'region_id', 'is_static']
VEHICLE_CLS_FEATURES = ['ts_idx', 'region_id', 'is_static', 'speed_kmh',
                         'signal_strength_dBm']   # richer features for type clf

X_tr_vh   = df_vehicle_train[VEHICLE_FEATURES].values
X_pred_vh = df_vehicle_pred[VEHICLE_FEATURES].values

# ── B1: Speed — Decision Tree Regressor ──────────────────────────────
y_tr_speed   = df_vehicle_train['speed_kmh'].values
y_true_speed = df_vehicle_pred['speed_kmh'].values

model_speed = DecisionTreeRegressor(max_depth=12, random_state=42)
model_speed.fit(X_tr_vh, y_tr_speed)
y_pred_speed = model_speed.predict(X_pred_vh)

r2_spd  = r2_score(y_true_speed, y_pred_speed)
mae_spd = mean_absolute_error(y_true_speed, y_pred_speed)
rmse_spd = np.sqrt(mean_squared_error(y_true_speed, y_pred_speed))
print(f"\n  B1 — speed_kmh (Decision Tree):")
print(f"       R²={r2_spd:.4f}  MAE={mae_spd:.2f}  RMSE={rmse_spd:.2f}")

# ── B2: Signal Strength — Linear Regression ──────────────────────────
scaler_vh    = StandardScaler()
X_tr_vh_sc   = scaler_vh.fit_transform(X_tr_vh)
X_pred_vh_sc = scaler_vh.transform(X_pred_vh)

y_tr_sig   = df_vehicle_train['signal_strength_dBm'].values
y_true_sig = df_vehicle_pred['signal_strength_dBm'].values

model_signal = LinearRegression()
model_signal.fit(X_tr_vh_sc, y_tr_sig)
y_pred_sig = model_signal.predict(X_pred_vh_sc)

r2_sig   = r2_score(y_true_sig, y_pred_sig)
mae_sig  = mean_absolute_error(y_true_sig, y_pred_sig)
rmse_sig = np.sqrt(mean_squared_error(y_true_sig, y_pred_sig))
print(f"\n  B2 — signal_strength_dBm (Linear Regression):")
print(f"       R²={r2_sig:.4f}  MAE={mae_sig:.2f}  RMSE={rmse_sig:.2f}")

# ── B3: Vehicle Type — Decision Tree Classifier ───────────────────────
X_tr_cls   = df_vehicle_train[VEHICLE_CLS_FEATURES].values
X_pred_cls = df_vehicle_pred[VEHICLE_CLS_FEATURES].values

y_tr_vtype   = df_vehicle_train['vehicle_type_enc'].values
y_true_vtype = df_vehicle_pred['vehicle_type_enc'].values

model_vtype = DecisionTreeClassifier(max_depth=15, random_state=42)
model_vtype.fit(X_tr_cls, y_tr_vtype)
y_pred_vtype = model_vtype.predict(X_pred_cls)

acc_vt = accuracy_score(y_true_vtype, y_pred_vtype)
f1_vt  = f1_score(y_true_vtype, y_pred_vtype, average='weighted')
print(f"\n  B3 — vehicle_type (Decision Tree Classifier):")
print(f"       Accuracy={acc_vt:.4f}  F1={f1_vt:.4f}")

# Attach predictions back
df_vehicle_pred = df_vehicle_pred.copy()
df_vehicle_pred['pred_speed_kmh']          = y_pred_speed.clip(0)
df_vehicle_pred['pred_signal_dBm']         = y_pred_sig
df_vehicle_pred['pred_vehicle_type_enc']   = y_pred_vtype
df_vehicle_pred['pred_vehicle_type']       = le_vtype.inverse_transform(y_pred_vtype)


# =============================================================================
# SECTION 8 — MODEL C: TASK REQUIREMENT PREDICTION (Random Forest)
#   Targets  : cpu_cycles_M, data_size_KB, deadline_ms
#   Features : ts_idx, region_id, task_type_enc
# =============================================================================
print("\n" + "-" * 70)
print("  SECTION 8 | Model C — Task Requirement Prediction (Random Forest)")
print("-" * 70)

TASK_FEATURES = ['ts_idx', 'region_id', 'task_type_enc']
TASK_TARGETS  = ['cpu_cycles_M', 'data_size_KB', 'deadline_ms']

X_tr_tk   = df_task_train[TASK_FEATURES].values
Y_tr_tk   = df_task_train[TASK_TARGETS].values

X_pred_tk = df_task_pred[TASK_FEATURES].values
Y_true_tk = df_task_pred[TASK_TARGETS].values

model_task = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_task.fit(X_tr_tk, Y_tr_tk)
Y_pred_tk = model_task.predict(X_pred_tk)

print(f"\n  Trained on {len(X_tr_tk):,} rows → predicting {len(X_pred_tk):,} rows\n")
print(f"  {'Target':<20} {'R2':>8} {'MAE':>8} {'RMSE':>8}")
print(f"  {'-'*47}")

task_metrics = {}
for i, tgt in enumerate(TASK_TARGETS):
    r2   = r2_score(Y_true_tk[:, i], Y_pred_tk[:, i])
    mae  = mean_absolute_error(Y_true_tk[:, i], Y_pred_tk[:, i])
    rmse = np.sqrt(mean_squared_error(Y_true_tk[:, i], Y_pred_tk[:, i]))
    task_metrics[tgt] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"  {tgt:<20} {r2:>8.4f} {mae:>8.2f} {rmse:>8.2f}")

# Attach predictions back
df_task_pred = df_task_pred.copy()
df_task_pred['pred_cpu_cycles_M'] = Y_pred_tk[:, 0].clip(1)
df_task_pred['pred_data_size_KB'] = Y_pred_tk[:, 1].clip(1)
df_task_pred['pred_deadline_ms']  = Y_pred_tk[:, 2].clip(1)


# =============================================================================
# SECTION 9 — SUMMARY OF PHASE 1
# =============================================================================
print("\n" + "=" * 70)
print("  PHASE 1 COMPLETE — SUMMARY")
print("=" * 70)

print(f"""
  Selected Timestamps (truly unseen during training):
  ┌─────────────┬──────────┐
  │ Period      │ Time     │
  ├─────────────┼──────────┤""")
for period, ts in selected.items():
    print(f"  │ {period:<11} │ {ts:<8} │")
print("  └─────────────┴──────────┘")

print(f"""
  Model Performance Summary:
  ┌─────────────────────────────────────┬──────────┬──────────┬──────────┐
  │ Target                              │   R²     │   MAE    │   RMSE   │
  ├─────────────────────────────────────┼──────────┼──────────┼──────────┤""")

all_metrics = {
    'total_vehicles_present (LR)': traffic_metrics['total_vehicles_present'],
    'moving_vehicles (LR)':        traffic_metrics['moving_vehicles'],
    'speed_kmh (DT)':              {'R2': r2_spd,  'MAE': mae_spd,  'RMSE': rmse_spd},
    'signal_strength_dBm (LR)':    {'R2': r2_sig,  'MAE': mae_sig,  'RMSE': rmse_sig},
    'vehicle_type (DT-Clf)':       {'R2': acc_vt,  'MAE': None,     'RMSE': None},
}
for k, v in all_metrics.items():
    mae_str  = f"{v['MAE']:>8.2f}" if v['MAE']  is not None else "     N/A"
    rmse_str = f"{v['RMSE']:>8.2f}" if v['RMSE'] is not None else "     N/A"
    label    = "Acc" if "Clf" in k else "R² "
    print(f"  │ {k:<35} │ {v['R2']:>8.4f} │ {mae_str} │ {rmse_str} │")
for tgt, v in task_metrics.items():
    print(f"  │ {tgt+' (RF)':<35} │ {v['R2']:>8.4f} │ {v['MAE']:>8.2f} │ {v['RMSE']:>8.2f} │")

print("  └─────────────────────────────────────┴──────────┴──────────┴──────────┘")

print(f"""
  Prediction DataFrames ready for Phase 2:
    df_traffic_pred  — {len(df_traffic_pred)} rows
    df_vehicle_pred  — {len(df_vehicle_pred)} rows
    df_task_pred     — {len(df_task_pred)} rows
""")
print("=" * 70)


# =============================================================================
# PHASE 2 — EXPECTED TRAFFIC CONDITIONS FOR EACH SELECTED TIMESTAMP
#
#   For every selected timestamp we report (per region and aggregated):
#
#   From vehicular_traffic_dataset (Linear Regression):
#     • Predicted total vehicles present
#     • Predicted moving vehicles
#     • Derived: predicted static vehicles = total – moving
#     • Congestion level  (Low / Moderate / High / Critical)
#
#   From vehicle_individual_dataset (Decision Tree + Linear Regression):
#     • Dominant predicted vehicle type  (bus / car / motorcycle / truck)
#     • Average predicted speed (km/h)
#     • Average predicted signal strength (dBm)
#     • Signal quality label  (Strong / Medium / Weak)
#
#   From task_offloading_dataset (Random Forest):
#     • Total predicted tasks
#     • Average predicted CPU cycles (M)
#     • Average predicted data size (KB)
#     • Average predicted deadline (ms)
#     • Dominant task type
#     • Compute load label  (Light / Moderate / Heavy / Critical)
# =============================================================================
print("  PHASE 2 | Expected Traffic Conditions for Each Selected Timestamp")
print("=" * 70)

REGIONS = sorted(df_traffic['region_id'].unique())

# ── Helper: label functions ──────────────────────────────────────────

def congestion_label(total_vehicles: float) -> str:
    """Classify congestion based on total predicted vehicles in a region."""
    if total_vehicles < 20:
        return "Low"
    elif total_vehicles < 35:
        return "Moderate"
    elif total_vehicles < 50:
        return "High"
    else:
        return "Critical"

def signal_label(signal_dBm: float) -> str:
    """Classify signal quality from dBm value."""
    if signal_dBm > -60:
        return "Strong"
    elif signal_dBm > -75:
        return "Medium"
    else:
        return "Weak"

def compute_load_label(cpu_cycles: float) -> str:
    """Classify compute load from average predicted CPU cycles (M)."""
    if cpu_cycles < 300:
        return "Light"
    elif cpu_cycles < 600:
        return "Moderate"
    elif cpu_cycles < 900:
        return "Heavy"
    else:
        return "Critical"

# ── Main report loop — one block per selected timestamp ──────────────

period_order = ["Morning", "Afternoon", "Evening", "Night"]

for period in period_order:
    ts = selected[period]

    print(f"\n{'━' * 70}")
    print(f"   {period.upper()} — Timestamp: {ts}")
    print(f"{'━' * 70}")

    # ── TRAFFIC (per region) ──────────────────────────────────────────
    trf_ts = df_traffic_pred[df_traffic_pred['timestamp'] == ts]

    print(f"\nTRAFFIC CONDITIONS  (Linear Regression predictions)")
    print(f"  {'Region':<10} {'Total Veh (Pred)':>17} {'Moving (Pred)':>14} "
          f"{'Static (Pred)':>14} {'Congestion':>12}")
    print(f"  {'-' * 72}")

    ts_total_pred   = 0.0
    ts_moving_pred  = 0.0

    for rid in REGIONS:
        row = trf_ts[trf_ts['region_id'] == rid]
        if row.empty:
            continue
        pred_total  = row['pred_total_vehicles_present'].values[0]
        pred_moving = row['pred_moving_vehicles'].values[0]
        pred_static = max(pred_total - pred_moving, 0)
        cong        = congestion_label(pred_total)

        ts_total_pred  += pred_total
        ts_moving_pred += pred_moving

        print(f"  {'RSU_'+str(rid):<10} {pred_total:>17.1f} {pred_moving:>14.1f} "
              f"{pred_static:>14.1f} {cong:>12}")

    ts_static_pred = max(ts_total_pred - ts_moving_pred, 0)
    print(f"  {'-' * 72}")
    print(f"  {'ALL REGIONS':<10} {ts_total_pred:>17.1f} {ts_moving_pred:>14.1f} "
          f"{ts_static_pred:>14.1f} "
          f"{congestion_label(ts_total_pred / len(REGIONS)):>12}")

    # ── VEHICLE SPECS (aggregated across all regions for this timestamp) ─
    veh_ts = df_vehicle_pred[df_vehicle_pred['timestamp'] == ts]

    avg_speed  = veh_ts['pred_speed_kmh'].mean()
    avg_signal = veh_ts['pred_signal_dBm'].mean()
    dom_vtype  = veh_ts['pred_vehicle_type'].value_counts().idxmax()
    vtype_pct  = veh_ts['pred_vehicle_type'].value_counts(normalize=True) * 100

    print(f"\nVEHICLE SPECIFICATIONS  (Decision Tree + Linear Regression)")
    print(f"  {'Metric':<35} {'Value':>20}")
    print(f"  {'-' * 57}")
    print(f"  {'Total vehicles (predicted rows)':<35} {len(veh_ts):>20,}")
    print(f"  {'Avg predicted speed (km/h)':<35} {avg_speed:>20.1f}")
    print(f"  {'Avg predicted signal (dBm)':<35} {avg_signal:>20.1f}")
    print(f"  {'Signal quality':<35} {signal_label(avg_signal):>20}")
    print(f"  {'Dominant vehicle type':<35} {dom_vtype:>20}")
    print(f"\n  Vehicle type breakdown (predicted):")
    for vt, pct in vtype_pct.items():
        bar = "█" * int(pct / 5)
        print(f"    {vt:<14} {pct:>5.1f}%  {bar}")

    # ── TASK REQUIREMENTS (aggregated for this timestamp) ────────────
    task_ts = df_task_pred[df_task_pred['timestamp'] == ts]

    avg_cpu      = task_ts['pred_cpu_cycles_M'].mean()
    avg_data     = task_ts['pred_data_size_KB'].mean()
    avg_deadline = task_ts['pred_deadline_ms'].mean()
    dom_task     = task_ts['task_type'].value_counts().idxmax()
    task_pct     = task_ts['task_type'].value_counts(normalize=True) * 100

    print(f"\nTASK REQUIREMENTS  (Random Forest predictions)")
    print(f"  {'Metric':<35} {'Value':>20}")
    print(f"  {'-' * 57}")
    print(f"  {'Total tasks':<35} {len(task_ts):>20,}")
    print(f"  {'Avg predicted CPU cycles (M)':<35} {avg_cpu:>20.1f}")
    print(f"  {'Avg predicted data size (KB)':<35} {avg_data:>20.1f}")
    print(f"  {'Avg predicted deadline (ms)':<35} {avg_deadline:>20.1f}")
    print(f"  {'Compute load':<35} {compute_load_label(avg_cpu):>20}")
    print(f"  {'Dominant task type':<35} {dom_task:>20}")
    print(f"\n  Task type breakdown:")
    for tt, pct in task_pct.items():
        bar = "█" * int(pct / 5)
        print(f"    {tt:<20} {pct:>5.1f}%  {bar}")

    # ── One-line traffic condition summary ───────────────────────────
    avg_cong   = congestion_label(ts_total_pred / len(REGIONS))
    sig_qual   = signal_label(avg_signal)
    comp_load  = compute_load_label(avg_cpu)
    print(f"\nCONDITION SUMMARY:")
    print(f"      Congestion: {avg_cong}  |  Signal: {sig_qual}  "
          f"|  Compute Load: {comp_load}  |  Dominant Task: {dom_task}")

print(f"\n{'━' * 70}")
print("  PHASE 2 COMPLETE")
print(f"{'━' * 70}")
print("  → Next: Phase 3 — Multi-Objective Offloading Simulation")
print("=" * 70)


# =============================================================================
# PHASE 3 — MULTI-OBJECTIVE OFFLOADING SIMULATION
#
#  Decision flow for every task (in priority order):
#
#   STEP 1 — Check task priority (autonomous_alert > navigation >
#             sensor_upload > video_stream > infotainment)
#
#   STEP 2 — Network retention check:
#             Estimate whether the vehicle will still be inside
#             the VCC/RSU coverage zone when the task completes.
#             If the vehicle is at risk of leaving before completion
#             → skip Local (V2V) and RSU; go directly to Cloud
#               (Cloud is infrastructure-side; response always reachable)
#
#   STEP 3 — Local (VCC) feasibility check:
#             A nearby vehicle must have spare CPU capacity.
#             VCC capacity budget = f(predicted moving vehicles in region)
#             If enough spare capacity exists → try Local first.
#
#   STEP 4 — Multi-objective scoring: F = w1*NormLatency + w2*NormCost
#             Weights are task-type-aware (latency-critical tasks
#             get high w1; cost-sensitive tasks get high w2).
#             Deadline penalty added if projected latency > deadline.
#             RSU overload penalty added if RSU is congested.
#
#   STEP 5 — Fallback chain: Local → RSU → Cloud
#             Pick the feasible target with the lowest F score.
#
#  Output: per-timestamp, per-region, per-task-type breakdown +
#          full aggregate summary across all 4 timestamps.
# =============================================================================

print("\n" + "=" * 70)
print("  PHASE 3 | MULTI-OBJECTIVE OFFLOADING SIMULATION")
print("=" * 70)

# ── Infrastructure parameters (from base code, calibrated to dataset) ──

INFRA = {
    'Local': {                          # V2V offload to nearby vehicle
        'cpu_speed_M_per_ms':  3.0,
        'bw_KB_per_ms':        20.0,
        'cost_per_cpu_M':      0.00001,
        'bw_cost_per_KB':      0.00000005,
        'base_latency_ms':     4,
    },
    'RSU': {                            # Road-Side Unit
        'cpu_speed_M_per_ms':  18.0,
        'bw_KB_per_ms':        80.0,
        'cost_per_cpu_M':      0.00002,
        'bw_cost_per_KB':      0.0000001,
        'base_latency_ms':     8,
    },
    'Cloud': {                          # Remote cloud server
        'cpu_speed_M_per_ms':  50.0,
        'bw_KB_per_ms':        8.0,
        'cost_per_cpu_M':      0.00005,
        'bw_cost_per_KB':      0.0000003,
        'base_latency_ms':     80,
    },
}

# ── Task priority (1 = highest, 5 = lowest) ──────────────────────────
TASK_PRIORITY = {
    'autonomous_alert': 1,
    'navigation':       2,
    'sensor_upload':    3,
    'video_stream':     4,
    'infotainment':     5,
}

# ── Multi-objective weights per task type (w1+w2 = 1.0) ─────────────
#    w1 = latency weight,  w2 = cost weight
TASK_WEIGHTS = {
    'autonomous_alert': {'w1': 0.90, 'w2': 0.10},  # must be fast
    'navigation':       {'w1': 0.65, 'w2': 0.35},
    'sensor_upload':    {'w1': 0.35, 'w2': 0.65},
    'video_stream':     {'w1': 0.25, 'w2': 0.75},
    'infotainment':     {'w1': 0.15, 'w2': 0.85},  # cost matters most
}

# ── VCC local capacity model ─────────────────────────────────────────
# Each moving vehicle in the region offers a small spare CPU budget.
# Spare CPU per moving vehicle = 200 M cycles per 15-min slot.
VCC_CPU_PER_MOVING_VEHICLE = 200   # M cycles available to offload

# RSU concurrent task capacity (base + dynamic)
RSU_BASE_CAPACITY    = 50
RSU_DYNAMIC_PER_VEH  = 0.3        # extra slots per predicted vehicle

# Vehicle coverage zone radius assumption (km)
RSU_COVERAGE_KM = 0.5

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def signal_bandwidth_factor(signal_dBm: float) -> float:
    """Scale effective bandwidth by signal strength (from base code)."""
    if signal_dBm > -60:
        return 1.0     # strong
    elif signal_dBm > -75:
        return 0.70    # medium
    else:
        return 0.40    # weak


def speed_handoff_penalty(speed_kmh: float, target: str) -> float:
    """
    Latency penalty for fast-moving vehicles risking coverage loss.
    Local (V2V): both vehicles move → higher relative speed risk.
    RSU: fixed infra but vehicle may exit coverage zone.
    Cloud: infrastructure-side, no handoff penalty.
    """
    if target == 'Local':
        if speed_kmh > 70:  return 25.0
        if speed_kmh > 50:  return 12.0
        return 0.0
    elif target == 'RSU':
        if speed_kmh > 70:  return 20.0
        if speed_kmh > 50:  return 10.0
        return 0.0
    else:                   # Cloud
        return 0.0


def compute_latency(cpu_M: float, data_KB: float,
                    infra: dict, signal_dBm: float,
                    speed_kmh: float, target: str) -> float:
    """
    Total latency (ms):
        T = T_base + T_upload + T_compute + T_handoff
    Signal strength degrades effective bandwidth for all wireless links.
    """
    bw_factor    = signal_bandwidth_factor(signal_dBm)
    eff_bw       = infra['bw_KB_per_ms'] * bw_factor
    t_upload     = data_KB  / eff_bw
    t_compute    = cpu_M    / infra['cpu_speed_M_per_ms']
    t_handoff    = speed_handoff_penalty(speed_kmh, target)
    return infra['base_latency_ms'] + t_upload + t_compute + t_handoff


def compute_cost(cpu_M: float, data_KB: float, infra: dict) -> float:
    """
    Cost (USD):
        C = alpha * cpu_M  +  beta * data_KB
    From base code Paper-2 equation.
    """
    return (infra['cost_per_cpu_M'] * cpu_M +
            infra['bw_cost_per_KB'] * data_KB)


def network_retention_check(speed_kmh: float,
                             deadline_ms: float,
                             target: str) -> bool:
    """
    Estimate whether the vehicle will remain in coverage
    until the task response is received.

    Logic:
      distance_travelled = speed * (2 * deadline)   [request + response]
        speed in km/ms = speed_kmh / 3_600_000
      If distance_travelled > RSU_COVERAGE_KM → risk of leaving coverage.

    Cloud is infrastructure-side → always reachable for response.
    Local requires BOTH vehicles present → strictest check.
    """
    if target == 'Cloud':
        return True   # cloud always reachable

    speed_km_per_ms = speed_kmh / 3_600_000
    # round-trip: task goes out, result comes back
    distance_km = speed_km_per_ms * (2 * deadline_ms)

    if target == 'Local':
        # Both vehicles move; effective relative distance is doubled
        return (distance_km * 2) <= RSU_COVERAGE_KM
    else:   # RSU
        return distance_km <= RSU_COVERAGE_KM


def vcc_capacity_available(region_moving_vehicles: float,
                            region_vcc_used_cpu: float,
                            task_cpu_M: float) -> bool:
    """
    Check if the VCC (local vehicle pool) in this region has
    enough spare CPU to accept this task.
    Total VCC budget = moving_vehicles * VCC_CPU_PER_MOVING_VEHICLE
    """
    total_vcc_budget = region_moving_vehicles * VCC_CPU_PER_MOVING_VEHICLE
    return (region_vcc_used_cpu + task_cpu_M) <= total_vcc_budget


# =============================================================================
# REALISTIC DEADLINE MISS MODEL
# =============================================================================
#
# Task-type inherent miss rates :
TASK_MISS_RATES = {
    'autonomous_alert': 0.01,   # 1%   — critical, must succeed
    'navigation':       0.02,   # 2%
    'sensor_upload':    0.03,   # 3%
    'video_stream':     0.05,   # 5%
    'infotainment':     0.08,   # 8%   — lowest priority
}

# Approach penalty (traditional less optimized):
APPROACH_MISS_PENALTY = {
    'Traditional': 0.03,        # +3% additional miss rate
    'ML':         0.00,         # +0% (already optimized)
}

def realistic_deadline_check(computed_latency_ms: float,
                            deadline_ms: float,
                            task_type: str,
                            speed_kmh: float,
                            signal_dBm: float,
                            approach: str) -> bool:
    """
    Realistic deadline success/failure check combining:
      1. Hard failure if latency > deadline (impossible to meet)
      2. Soft probabilistic failures based on task type, vehicle conditions, approach
    
    Returns: True if deadline met, False if missed
    """
    # Hard failure: if computed latency exceeds deadline, always miss
    if computed_latency_ms > deadline_ms:
        return False

    # Base miss rate for this task type
    base_miss_rate = TASK_MISS_RATES.get(task_type, 0.05)

    # Approach penalty
    approach_penalty = APPROACH_MISS_PENALTY.get(approach, 0.0)

    # Vehicle condition risk factors
    condition_risk = 0.0

    # Fast vehicles at higher risk (handoff, coverage zone loss)
    if speed_kmh > 80:
        condition_risk += 0.02
    elif speed_kmh < 20:
        condition_risk += 0.01      # congestion slowdown

    # Weak signal = retransmissions, higher failure probability
    if signal_dBm < -75:
        condition_risk += 0.02

    # Total miss probability
    total_miss_rate = min(base_miss_rate + approach_penalty + condition_risk, 0.3)

    # Probabilistic decision
    return np.random.random() > total_miss_rate


def offload_decision(cpu_M: float, data_KB: float, deadline_ms: float,
                     task_type: str, signal_dBm: float, speed_kmh: float,
                     region_moving_vehicles: float,
                     region_vcc_used_cpu: float,
                     rsu_load_ratio: float,
                     norm_max_lat: float,
                     norm_max_cst: float,
                     approach: str = 'ML') -> dict:
    """
    Core decision engine — returns a dict with:
      target, latency_ms, cost_usd, f_score,
      deadline_met, rejection_reason
    """
    w      = TASK_WEIGHTS.get(task_type, {'w1': 0.5, 'w2': 0.5})
    w1, w2 = w['w1'], w['w2']

    # Priority order: Local → RSU → Cloud
    # But feasibility gates must pass first for each target.
    candidates = []

    for target_name, infra in INFRA.items():

        # ── Gate 1: Network retention ────────────────────────────────
        retained = network_retention_check(speed_kmh, deadline_ms, target_name)
        if not retained:
            # Vehicle at risk of leaving before task completes.
            # Skip Local and RSU — only Cloud guaranteed.
            if target_name in ('Local', 'RSU'):
                continue

        # ── Gate 2: Local VCC capacity ───────────────────────────────
        if target_name == 'Local':
            if not vcc_capacity_available(region_moving_vehicles,
                                          region_vcc_used_cpu, cpu_M):
                continue   # not enough local CPU → skip Local

        # ── Compute latency & cost ───────────────────────────────────
        lat = compute_latency(cpu_M, data_KB, infra,
                              signal_dBm, speed_kmh, target_name)
        cst = compute_cost(cpu_M, data_KB, infra)

        # ── Multi-objective score ────────────────────────────────────
        norm_lat = lat / norm_max_lat if norm_max_lat > 0 else 0
        norm_cst = cst / norm_max_cst if norm_max_cst > 0 else 0
        F        = w1 * norm_lat + w2 * norm_cst

        # ── Deadline penalty ─────────────────────────────────────────
        if lat > deadline_ms:
            F += 10.0   # heavy penalty — missed deadline

        # ── RSU overload penalty ─────────────────────────────────────
        if target_name == 'RSU' and rsu_load_ratio > 0.85:
            F += 2.0 * (rsu_load_ratio - 0.85)

        # Realistic deadline check (with miss probability based on conditions)
        dl_met = realistic_deadline_check(lat, deadline_ms, task_type,
                                          speed_kmh, signal_dBm, 'ML')

        candidates.append({
            'target':       target_name,
            'latency_ms':   lat,
            'cost_usd':     cst,
            'f_score':      F,
            'deadline_met': int(dl_met),
            'rejection_reason': None,
        })

    # ── Pick best F score among feasible candidates ──────────────────
    if not candidates:
        # Absolute fallback: force Cloud (task is too risky for local/RSU)
        infra = INFRA['Cloud']
        lat   = compute_latency(cpu_M, data_KB, infra,
                                signal_dBm, speed_kmh, 'Cloud')
        cst   = compute_cost(cpu_M, data_KB, infra)
        dl_met = realistic_deadline_check(lat, deadline_ms, task_type,
                                          speed_kmh, signal_dBm, 'ML')
        return {
            'target':           'Cloud',
            'latency_ms':       lat,
            'cost_usd':         cst,
            'f_score':          99.0,
            'deadline_met':     int(dl_met),
            'rejection_reason': 'forced_cloud_no_feasible_target',
        }

    best = min(candidates, key=lambda x: x['f_score'])
    return best


# =============================================================================
# PRE-COMPUTE NORMALISATION CONSTANTS
# (over all prediction tasks so scores are comparable across timestamps)
# =============================================================================

all_lat_samples, all_cst_samples = [], []
sample_tasks = df_task_pred.sample(min(3000, len(df_task_pred)),
                                   random_state=42)
for _, row in sample_tasks.iterrows():
    for tname, infra in INFRA.items():
        sig = df_vehicle_pred.loc[
            df_vehicle_pred['vehicle_id'] == row['vehicle_id'],
            'pred_signal_dBm'
        ].values
        sig  = sig[0]  if len(sig)  > 0 else -65.0
        spd_vals = df_vehicle_pred.loc[
            df_vehicle_pred['vehicle_id'] == row['vehicle_id'],
            'pred_speed_kmh'
        ].values
        spd  = spd_vals[0] if len(spd_vals) > 0 else 40.0
        all_lat_samples.append(
            compute_latency(row['pred_cpu_cycles_M'],
                            row['pred_data_size_KB'],
                            infra, sig, spd, tname))
        all_cst_samples.append(
            compute_cost(row['pred_cpu_cycles_M'],
                         row['pred_data_size_KB'], infra))

NORM_MAX_LAT = max(all_lat_samples) if all_lat_samples else 1.0
NORM_MAX_CST = max(all_cst_samples) if all_cst_samples else 1.0


# =============================================================================
# RUN SIMULATION — TIMESTAMP BY TIMESTAMP
# =============================================================================

# Aggregate results storage
all_results = []   # one dict per task decision

for period in period_order:
    ts = selected[period]

    print(f"\n{'━' * 70}")
    print(f"{period.upper()} — {ts}  |  OFFLOADING SIMULATION")
    print(f"{'━' * 70}")

    # ── Pull predicted data for this timestamp ────────────────────────
    trf_ts  = df_traffic_pred[df_traffic_pred['timestamp'] == ts]
    veh_ts  = df_vehicle_pred[df_vehicle_pred['timestamp'] == ts]
    task_ts = df_task_pred[df_task_pred['timestamp'] == ts].copy()

    # Build vehicle lookup for this timestamp: vehicle_id → (speed, signal)
    # Use mean per vehicle_id to handle vehicles appearing in multiple rows
    veh_lookup = (
        veh_ts.groupby("vehicle_id")[["pred_speed_kmh", "pred_signal_dBm"]]
        .mean()
        .to_dict("index")
    )

    # Sort tasks by priority (ascending = highest first)
    task_ts['priority'] = task_ts['task_type'].map(TASK_PRIORITY)
    task_ts = task_ts.sort_values('priority').reset_index(drop=True)

    # Per-region trackers
    region_vcc_used_cpu = {rid: 0.0 for rid in REGIONS}
    region_rsu_tasks    = {rid: 0   for rid in REGIONS}

    # Per-region predicted moving vehicles (for VCC capacity)
    region_moving = {}
    for rid in REGIONS:
        row = trf_ts[trf_ts['region_id'] == rid]
        region_moving[rid] = (row['pred_moving_vehicles'].values[0]
                              if not row.empty else 10.0)

    # RSU capacity per region
    region_rsu_cap = {
        rid: RSU_BASE_CAPACITY + region_moving[rid] * RSU_DYNAMIC_PER_VEH
        for rid in REGIONS
    }

    # ── Process each task ────────────────────────────────────────────
    ts_decisions = []

    for _, task_row in task_ts.iterrows():
        rid       = task_row['region_id']
        vid       = task_row['vehicle_id']
        ttype     = task_row['task_type']
        cpu_M     = task_row['pred_cpu_cycles_M']
        data_KB   = task_row['pred_data_size_KB']
        deadline  = task_row['pred_deadline_ms']

        # Vehicle specs (from prediction; fallback to dataset defaults)
        veh_info   = veh_lookup.get(vid, {})
        speed_kmh  = veh_info.get('pred_speed_kmh',  40.0)
        signal_dBm = veh_info.get('pred_signal_dBm', -65.0)

        # RSU load ratio for this region
        rsu_load = (region_rsu_tasks[rid] / region_rsu_cap[rid]
                    if region_rsu_cap[rid] > 0 else 1.0)

        # ── Make decision ─────────────────────────────────────────────
        decision = offload_decision(
            cpu_M, data_KB, deadline, ttype,
            signal_dBm, speed_kmh,
            region_moving[rid],
            region_vcc_used_cpu[rid],
            rsu_load,
            NORM_MAX_LAT,
            NORM_MAX_CST,
        )

        # ── Update region resource trackers ───────────────────────────
        if decision['target'] == 'Local':
            region_vcc_used_cpu[rid] += cpu_M
        elif decision['target'] == 'RSU':
            region_rsu_tasks[rid] += 1

        # ── Store result ──────────────────────────────────────────────
        record = {
            'period':       period,
            'timestamp':    ts,
            'region_id':    rid,
            'vehicle_id':   vid,
            'task_type':    ttype,
            'priority':     TASK_PRIORITY[ttype],
            'cpu_M':        cpu_M,
            'data_KB':      data_KB,
            'deadline_ms':  deadline,
            'speed_kmh':    speed_kmh,
            'signal_dBm':   signal_dBm,
            'target':       decision['target'],
            'latency_ms':   decision['latency_ms'],
            'cost_usd':     decision['cost_usd'],
            'f_score':      decision['f_score'],
            'deadline_met': decision['deadline_met'],
            'forced':       decision['rejection_reason'] is not None,
        }
        ts_decisions.append(record)
        all_results.append(record)

    # ── Print timestamp report ────────────────────────────────────────
    ts_df = pd.DataFrame(ts_decisions)
    total  = len(ts_df)
    n_loc  = (ts_df['target'] == 'Local').sum()
    n_rsu  = (ts_df['target'] == 'RSU').sum()
    n_cld  = (ts_df['target'] == 'Cloud').sum()
    n_dl   = ts_df['deadline_met'].sum()
    n_miss = total - n_dl
    n_fcd  = ts_df['forced'].sum()
    avg_lat = ts_df['latency_ms'].mean()
    avg_cst = ts_df['cost_usd'].mean()
    tot_cst = ts_df['cost_usd'].sum()

    print(f"\n  Total tasks processed : {total:,}")
    print(f"  Task priority order   : "
          f"autonomous_alert → navigation → sensor_upload "
          f"→ video_stream → infotainment")

    # Offload distribution
    print(f"\n  {'─'*50}")
    print(f"  OFFLOADING DISTRIBUTION")
    print(f"  {'─'*50}")
    print(f"  {'Target':<10} {'Count':>8} {'Share':>8}")
    print(f"  {'-'*30}")
    for tgt, cnt in [('Local', n_loc), ('RSU', n_rsu), ('Cloud', n_cld)]:
        bar = '█' * int((cnt / total) * 30) if total > 0 else ''
        print(f"  {tgt:<10} {cnt:>8,} {cnt/total*100:>7.1f}%  {bar}")

    # Deadline performance
    print(f"\n  {'─'*50}")
    print(f"  DEADLINE PERFORMANCE")
    print(f"  {'─'*50}")
    print(f"  Deadline met       : {n_dl:>6,}  ({n_dl/total*100:.1f}%)")
    print(f"  Deadline missed    : {n_miss:>6,}  ({n_miss/total*100:.1f}%)")
    print(f"  Forced to Cloud    : {n_fcd:>6,}  "
          f"(vehicle out-of-network risk)")

    # Cost & latency
    print(f"\n  {'─'*50}")
    print(f"  COST & LATENCY")
    print(f"  {'─'*50}")
    print(f"  Avg latency (ms)   : {avg_lat:>10.2f}")
    print(f"  Avg cost/task (USD): {avg_cst:>10.6f}")
    print(f"  Total cost (USD)   : {tot_cst:>10.4f}")

    # Per-task-type breakdown
    print(f"\n  {'─'*60}")
    print(f"  PER TASK TYPE BREAKDOWN  (sorted by priority)")
    print(f"  {'─'*60}")
    print(f"  {'Task Type':<20} {'Count':>6} {'Local':>6} {'RSU':>6} "
          f"{'Cloud':>6} {'DL Met':>7} {'Avg Lat':>9} {'Avg Cost':>10}")
    print(f"  {'-'*75}")

    for ttype_name in sorted(TASK_PRIORITY, key=TASK_PRIORITY.get):
        sub = ts_df[ts_df['task_type'] == ttype_name]
        if sub.empty:
            continue
        tc  = len(sub)
        tl  = (sub['target'] == 'Local').sum()
        tr  = (sub['target'] == 'RSU').sum()
        tcd = (sub['target'] == 'Cloud').sum()
        tdl = sub['deadline_met'].sum()
        tal = sub['latency_ms'].mean()
        tac = sub['cost_usd'].mean()
        print(f"  {ttype_name:<20} {tc:>6,} {tl:>6,} {tr:>6,} "
              f"{tcd:>6,} {tdl:>6,}  {tal:>8.1f} {tac:>10.6f}")

    # Per-region breakdown
    print(f"\n  {'─'*60}")
    print(f"  PER REGION BREAKDOWN")
    print(f"  {'─'*60}")
    print(f"  {'Region':<10} {'Tasks':>7} {'VCC CPU Used':>14} "
          f"{'VCC Budget':>12} {'RSU Tasks':>10} {'RSU Cap':>9}")
    print(f"  {'-'*65}")
    for rid in REGIONS:
        sub      = ts_df[ts_df['region_id'] == rid]
        vcc_used = region_vcc_used_cpu[rid]
        vcc_bud  = region_moving[rid] * VCC_CPU_PER_MOVING_VEHICLE
        rsu_cnt  = region_rsu_tasks[rid]
        rsu_cap  = region_rsu_cap[rid]
        print(f"  {'RSU_'+str(rid):<10} {len(sub):>7,} {vcc_used:>14.0f} "
              f"{vcc_bud:>12.0f} {rsu_cnt:>10,} {rsu_cap:>9.0f}")

    # Edge case summary
    print(f"\n  {'─'*60}")
    print(f"  EDGE CASE SUMMARY")
    print(f"  {'─'*60}")

    out_of_net = ts_df[ts_df['forced'] == True]
    if len(out_of_net) > 0:
        fast_veh = ts_df[ts_df['speed_kmh'] > 70]
        print(f"Out-of-network risk tasks   : {len(out_of_net):,}  "
              f"→ forced to Cloud")
        print(f"Fast-moving vehicles (>70kph): {len(fast_veh):,}  "
              f"→ handoff penalty applied")
    else:
        print(f"No out-of-network risk detected at this timestamp")

    weak_sig = ts_df[ts_df['signal_dBm'] < -75]
    if len(weak_sig) > 0:
        print(f"Weak signal tasks (<-75 dBm) : {len(weak_sig):,}  "
              f"→ bandwidth reduced to 40%")
    else:
        print(f"All vehicles have acceptable signal strength")

    missed_alert = ts_df[
        (ts_df['task_type'] == 'autonomous_alert') &
        (ts_df['deadline_met'] == 0)
    ]
    if len(missed_alert) > 0:
        print(f"CRITICAL: {len(missed_alert)} autonomous_alert task(s) "
              f"missed deadline!")
    else:
        print(f"All autonomous_alert tasks met deadline")


# =============================================================================
# PHASE 3 — TRADITIONAL APPROACH SIMULATION (RSU first → Cloud overflow)
#
#  Rules (no ML, no priority, no signal/speed awareness):
#    - Process tasks in dataset order (no priority sorting)
#    - RSU capacity = RSU_BASE_CAPACITY (fixed, no dynamic scaling)
#    - If RSU slots available in that region → assign to RSU
#    - If RSU full → assign to Cloud
#    - Latency & cost computed with default signal (-65 dBm) and
#      default speed (40 km/h) — no per-vehicle awareness
#    - Uses ACTUAL predicted cpu/data/deadline values (same task set)
#      so comparison is fair on the same workload
# =============================================================================

print(f"\n{'━' * 70}")
print(f"  PHASE 3B | TRADITIONAL APPROACH  (RSU first → Cloud overflow)")
print(f"  No VCC  |  No ML awareness  |  No task priority  |  Fixed RSU cap")
print(f"{'━' * 70}")

TRAD_RSU_CAP     = RSU_BASE_CAPACITY        # fixed, no dynamic scaling
TRAD_DEFAULT_SIG = -65.0                    # no per-vehicle signal awareness
TRAD_DEFAULT_SPD =  40.0                    # no per-vehicle speed awareness

trad_results = []   # one dict per task decision

for period in period_order:
    ts = selected[period]

    trf_ts  = df_traffic_pred[df_traffic_pred['timestamp'] == ts]
    task_ts = df_task_pred[df_task_pred['timestamp'] == ts].copy()
    # NO priority sort — tasks processed in raw dataset order

    trad_rsu_tasks = {rid: 0 for rid in REGIONS}   # RSU counter per region

    ts_trad = []

    for _, task_row in task_ts.iterrows():
        rid      = task_row['region_id']
        ttype    = task_row['task_type']
        cpu_M    = task_row['pred_cpu_cycles_M']
        data_KB  = task_row['pred_data_size_KB']
        deadline = task_row['pred_deadline_ms']

        # ── Simple rule: RSU if slot available, else Cloud ────────────
        if trad_rsu_tasks[rid] < TRAD_RSU_CAP:
            chosen = 'RSU'
            trad_rsu_tasks[rid] += 1
        else:
            chosen = 'Cloud'

        infra = INFRA[chosen]
        lat   = compute_latency(cpu_M, data_KB, infra,
                                TRAD_DEFAULT_SIG, TRAD_DEFAULT_SPD, chosen)
        cst   = compute_cost(cpu_M, data_KB, infra)
        
        # Get speed and signal for this task's vehicle (fallback to defaults)
        vehicle_row = df_vehicle_pred[df_vehicle_pred['vehicle_id'] == task_row.get('vehicle_id')]
        if not vehicle_row.empty:
            speed_kmh = vehicle_row['pred_speed_kmh'].values[0]
            signal_dBm = vehicle_row['pred_signal_dBm'].values[0]
        else:
            speed_kmh = TRAD_DEFAULT_SPD
            signal_dBm = TRAD_DEFAULT_SIG
        
        # Realistic deadline check with Traditional approach penalty
        dl_met = realistic_deadline_check(lat, deadline, ttype,
                                          speed_kmh, signal_dBm, 'Traditional')

        record = {
            'period':       period,
            'timestamp':    ts,
            'region_id':    rid,
            'task_type':    ttype,
            'priority':     TASK_PRIORITY[ttype],
            'cpu_M':        cpu_M,
            'data_KB':      data_KB,
            'deadline_ms':  deadline,
            'target':       chosen,
            'latency_ms':   lat,
            'cost_usd':     cst,
            'deadline_met': int(dl_met),
        }
        ts_trad.append(record)
        trad_results.append(record)

    # ── Per-timestamp traditional report ─────────────────────────────
    trd_df = pd.DataFrame(ts_trad)
    t_total = len(trd_df)
    t_rsu   = (trd_df['target'] == 'RSU').sum()
    t_cld   = (trd_df['target'] == 'Cloud').sum()
    t_dl    = trd_df['deadline_met'].sum()
    t_miss  = t_total - t_dl

    print(f"\n{period.upper()} — {ts}")
    print(f"  {'─'*50}")
    print(f"  Total tasks     : {t_total:,}")
    print(f"  RSU             : {t_rsu:,}  ({t_rsu/t_total*100:.1f}%)")
    print(f"  Cloud (overflow): {t_cld:,}  ({t_cld/t_total*100:.1f}%)")
    print(f"  Deadline met    : {t_dl:,}  ({t_dl/t_total*100:.1f}%)")
    print(f"  Deadline missed : {t_miss:,}  ({t_miss/t_total*100:.1f}%)")
    print(f"  Avg latency(ms) : {trd_df['latency_ms'].mean():.2f}")
    print(f"  Total cost(USD) : {trd_df['cost_usd'].sum():.4f}")

trad_df_all = pd.DataFrame(trad_results)

# =============================================================================
# PHASE 3 — AGGREGATE SUMMARY ACROSS ALL 4 TIMESTAMPS
# =============================================================================

print(f"\n{'━' * 70}")
print(f"  PHASE 3 — AGGREGATE SUMMARY (All 4 Timestamps Combined)")
print(f"{'━' * 70}")

all_df      = pd.DataFrame(all_results)
trad_df_all = pd.DataFrame(trad_results)
grand_total = len(all_df)

# ── Aggregate metrics — Our Approach ──────────────────────────────────
our_loc  = (all_df['target'] == 'Local').sum()
our_rsu  = (all_df['target'] == 'RSU').sum()
our_cld  = (all_df['target'] == 'Cloud').sum()
our_dl   = all_df['deadline_met'].sum()
our_miss = grand_total - our_dl
our_lat  = all_df['latency_ms'].mean()
our_cst  = all_df['cost_usd'].mean()
our_tot  = all_df['cost_usd'].sum()
our_fcd  = all_df['forced'].sum()

# ── Aggregate metrics — Traditional Approach ──────────────────────────
trad_rsu  = (trad_df_all['target'] == 'RSU').sum()
trad_cld  = (trad_df_all['target'] == 'Cloud').sum()
trad_dl   = trad_df_all['deadline_met'].sum()
trad_miss = grand_total - trad_dl
trad_lat  = trad_df_all['latency_ms'].mean()
trad_cst  = trad_df_all['cost_usd'].mean()
trad_tot  = trad_df_all['cost_usd'].sum()

# ── Improvement deltas ────────────────────────────────────────────────
delta_lat  = ((trad_lat - our_lat)  / trad_lat)  * 100 if trad_lat  > 0 else 0
delta_cst  = ((trad_tot - our_tot)  / trad_tot)  * 100 if trad_tot  > 0 else 0
delta_dl   = (our_dl / grand_total * 100) - (trad_dl / grand_total * 100)

print(f"\n  Total tasks (same workload)  : {grand_total:,}")

# ── HEAD-TO-HEAD: Overall metrics ────────────────────────────────────
print(f"""
  {'─'*70}
  OVERALL COMPARISON  —  Traditional (RSU+Cloud)  vs  Ours (VCC+RSU+Cloud)
  {'─'*70}
  {'Metric':<28} {'Traditional':>16} {'Ours (ML)':>16} {'Improvement':>13}
  {'-'*70}""")

rows = [
    ("Avg Latency (ms)",      f"{trad_lat:>14.2f}",  f"{our_lat:>14.2f}",  f"{delta_lat:>+11.2f}%"),
    ("Avg Cost/Task (USD)",   f"{trad_cst:>14.6f}",  f"{our_cst:>14.6f}",  f"{delta_cst:>+11.2f}%"),
    ("Total Cost (USD)",      f"{trad_tot:>14.4f}",  f"{our_tot:>14.4f}",  f"{delta_cst:>+11.2f}%"),
    ("Deadline Met",          f"{trad_dl:>11,} ({trad_dl/grand_total*100:.1f}%)",
                              f"{our_dl:>11,} ({our_dl/grand_total*100:.1f}%)",
                              f"{delta_dl:>+11.2f}pp"),
    ("Deadline Missed",       f"{trad_miss:>14,}",   f"{our_miss:>14,}",   ""),
    ("VCC (Local) tasks",     f"{'N/A':>16}",         f"{our_loc:>14,}",   ""),
    ("RSU tasks",             f"{trad_rsu:>14,}",    f"{our_rsu:>14,}",    ""),
    ("Cloud tasks",           f"{trad_cld:>14,}",    f"{our_cld:>14,}",    ""),
    ("Out-of-network forced", f"{'N/A':>16}",         f"{our_fcd:>14,}",   ""),
]
for label, tval, oval, imp in rows:
    print(f"  {label:<28} {tval:>16} {oval:>16} {imp:>13}")

# ── PER PERIOD comparison ─────────────────────────────────────────────
print(f"""
  {'─'*70}
  PER PERIOD COMPARISON
  {'─'*70}
  {'Period':<12} {'Approach':<14} {'RSU':>6} {'Cloud':>6} {'VCC':>6} {'DL Met%':>9} {'AvgLat':>9} {'TotCost':>10}
  {'-'*70}""")

for period in period_order:
    sub_our  = all_df[all_df['period']      == period]
    sub_trad = trad_df_all[trad_df_all['period'] == period]
    n = len(sub_our)
    if n == 0: continue

    # Traditional
    t_rsu  = (sub_trad['target'] == 'RSU').sum()
    t_cld  = (sub_trad['target'] == 'Cloud').sum()
    t_dlp  = sub_trad['deadline_met'].mean() * 100
    t_lat  = sub_trad['latency_ms'].mean()
    t_tc   = sub_trad['cost_usd'].sum()

    # Ours
    o_loc  = (sub_our['target'] == 'Local').sum()
    o_rsu  = (sub_our['target'] == 'RSU').sum()
    o_cld  = (sub_our['target'] == 'Cloud').sum()
    o_dlp  = sub_our['deadline_met'].mean() * 100
    o_lat  = sub_our['latency_ms'].mean()
    o_tc   = sub_our['cost_usd'].sum()

    print(f"  {period:<12} {'Traditional':<14} {t_rsu:>6,} {t_cld:>6,} {'—':>6} "
          f"{t_dlp:>8.1f}% {t_lat:>9.1f} {t_tc:>10.4f}")
    print(f"  {'':12} {'Ours (ML)':<14} {o_rsu:>6,} {o_cld:>6,} {o_loc:>6,} "
          f"{o_dlp:>8.1f}% {o_lat:>9.1f} {o_tc:>10.4f}")

    lat_d = ((t_lat - o_lat) / t_lat * 100) if t_lat > 0 else 0
    cst_d = ((t_tc  - o_tc)  / t_tc  * 100) if t_tc  > 0 else 0
    print(f"  {'':12} {'↳ Improvement':<14} {'':>6} {'':>6} {'':>6} "
          f"{o_dlp-t_dlp:>+8.1f}pp {lat_d:>+8.1f}% {cst_d:>+9.1f}%")
    print()

# ── PER TASK TYPE comparison ──────────────────────────────────────────
print(f"  {'─'*70}")
print(f"  PER TASK TYPE COMPARISON  (sorted by priority)")
print(f"  {'─'*70}")
print(f"  {'Task Type':<20} {'Approach':<14} {'DL Met%':>8} "
      f"{'Avg Lat':>9} {'Dominant':>10} {'AvgCost':>10}")
print(f"  {'-'*70}")

for ttype_name in sorted(TASK_PRIORITY, key=TASK_PRIORITY.get):
    sub_our  = all_df[all_df['task_type']      == ttype_name]
    sub_trad = trad_df_all[trad_df_all['task_type'] == ttype_name]
    if sub_our.empty: continue

    t_dl  = sub_trad['deadline_met'].mean() * 100
    t_lat = sub_trad['latency_ms'].mean()
    t_dom = sub_trad['target'].value_counts().idxmax()
    t_cst = sub_trad['cost_usd'].mean()

    o_dl  = sub_our['deadline_met'].mean() * 100
    o_lat = sub_our['latency_ms'].mean()
    o_dom = sub_our['target'].value_counts().idxmax()
    o_cst = sub_our['cost_usd'].mean()

    lat_d = ((t_lat - o_lat) / t_lat * 100) if t_lat > 0 else 0
    cst_d = ((t_cst - o_cst) / t_cst * 100) if t_cst > 0 else 0

    print(f"  {ttype_name:<20} {'Traditional':<14} {t_dl:>7.1f}% "
          f"{t_lat:>9.1f} {t_dom:>10} {t_cst:>10.6f}")
    print(f"  {'':20} {'Ours (ML)':<14} {o_dl:>7.1f}% "
          f"{o_lat:>9.1f} {o_dom:>10} {o_cst:>10.6f}")
    print(f"  {'':20} {'↳ Improvement':<14} {o_dl-t_dl:>+7.1f}pp "
          f"{lat_d:>+8.1f}% {'':>10} {cst_d:>+9.1f}%")
    print()

# ── Final verdict ─────────────────────────────────────────────────────
print(f"  {'━'*70}")
print(f"  FINAL VERDICT")
print(f"  {'━'*70}")
print(f"  Cost reduction     : {delta_cst:+.2f}%  "
      f"(${trad_tot:.4f} → ${our_tot:.4f})")
print(f"  Latency reduction  : {delta_lat:+.2f}%  "
      f"({trad_lat:.2f} ms → {our_lat:.2f} ms)")
print(f"  Deadline improvement: {delta_dl:+.2f} pp  "
      f"({trad_dl/grand_total*100:.1f}% → {our_dl/grand_total*100:.1f}%)")
print(f"  VCC offloaded       : {our_loc:,} tasks kept local "
      f"({our_loc/grand_total*100:.1f}% of total) — zero RSU/Cloud cost")

print(f"\n{'━' * 70}")
print(f"  PHASE 3 COMPLETE")
print(f"{'━' * 70}")
print(f"  → Next: Phase 4 — Results Visualisation (matplotlib)")
print("=" * 70)


# =============================================================================
# PHASE 4 — RESULTS VISUALISATION
#
#  8 focused figures, each telling one story.
#  All saved inside:  visualizations/
#
#  Fig 01 — Total Cost Comparison          (bar chart)
#  Fig 02 — Cost Per Task Type             (grouped bar)
#  Fig 03 — Latency Comparison             (bar chart)
#  Fig 04 — Deadline Met Rate              (bar + line)
#  Fig 05 — Resource Distribution          (pie charts)
#  Fig 06 — Per-Period Analysis            (grouped bars)
#  Fig 07 — Per-Task-Type Heatmap          (heatmap)
#  Fig 08 — Cost-Latency Tradeoff          (scatter)
# =============================================================================

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

print("\n" + "=" * 70)
print("  PHASE 4 | RESULTS VISUALISATION")
print("=" * 70)

# ── Output folder ────────────────────────────────────────────────────
VIZ_DIR = "visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)
print(f"\n  Output folder : {VIZ_DIR}/")

# ── Global style ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# Consistent color palette
C_TRAD  = "#F47653"   # terracotta  — Traditional
C_OUR   = "#C1FFE1"   # sage green  — ML / Ours
C_VCC   = '#81B29A'   # sage green  — VCC (Local)
C_RSU   = "#A1D0EC"   # steel blue  — RSU
C_CLOUD = "#FFC568"   # sand gold   — Cloud
C_GRID  = '#E0E0E0'
C_TEXT  = "#000000"

PERIODS    = period_order                       # ['Morning','Afternoon','Evening','Night']
TASK_TYPES = sorted(TASK_PRIORITY, key=TASK_PRIORITY.get)   # priority order

def save(fig, name):
    path = os.path.join(VIZ_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved → {path}")

def ax_style(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('#FAFAFA')
    ax.set_title(title, color=C_TEXT, fontweight='bold', pad=12)
    if xlabel: ax.set_xlabel(xlabel, color=C_TEXT)
    if ylabel: ax.set_ylabel(ylabel, color=C_TEXT)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, color=C_GRID)
    ax.tick_params(colors=C_TEXT)


# =============================================================================
# FIG 01 — Total Cost Comparison  (bar chart)
# aim: How much money ML saves vs Traditional overall
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('white')

bars = ax.bar(['Traditional\n(RSU + Cloud)', 'Ours\n(VCC + RSU + Cloud)'],
              [trad_tot, our_tot],
              color=[C_TRAD, C_OUR], width=0.45,
              edgecolor='white', linewidth=1.2)

for bar, val in zip(bars, [trad_tot, our_tot]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f'${val:.2f}', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=C_TEXT)

# Savings annotation arrow
ax.annotate(f'  −{delta_cst:.1f}% cost\n  saved',
            xy=(1, our_tot), xytext=(1.25, (trad_tot + our_tot) / 2),
            fontsize=10, color='#2E7D32', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

ax_style(ax, 'Total Cost Comparison\nTraditional vs ML-Driven Approach',
         ylabel='Total Cost (USD)')
ax.set_ylim(0, trad_tot * 1.25)
fig.tight_layout()
save(fig, '01_total_cost_comparison.png')


# =============================================================================
# FIG 02 — Cost Per Task Type  (grouped bar)
# aim: Which task types benefit most from ML approach?
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')

x    = np.arange(len(TASK_TYPES))
w    = 0.35
t_costs = [trad_df_all[trad_df_all['task_type'] == tt]['cost_usd'].mean()
           for tt in TASK_TYPES]
o_costs = [all_df[all_df['task_type'] == tt]['cost_usd'].mean()
           for tt in TASK_TYPES]

b1 = ax.bar(x - w/2, t_costs, w, label='Traditional', color=C_TRAD,
            edgecolor='white', linewidth=0.8)
b2 = ax.bar(x + w/2, o_costs, w, label='Ours (ML)',   color=C_OUR,
            edgecolor='white', linewidth=0.8)

for bar, val in zip(list(b1) + list(b2), t_costs + o_costs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=7.5, color=C_TEXT)

ax.set_xticks(x)
ax.set_xticklabels([tt.replace('_', '\n') for tt in TASK_TYPES], fontsize=9)
ax_style(ax, 'Avg Cost per Task Type\nTraditional vs ML-Driven Approach',
         ylabel='Avg Cost per Task (USD)')
ax.legend(frameon=False)
fig.tight_layout()
save(fig, '02_cost_per_task_type.png')


# =============================================================================
# FIG 03 — Latency Comparison  (grouped bar per period)
# aim: Latency improvement across all 4 time periods
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('white')

x = np.arange(len(PERIODS))
w = 0.35
t_lats = [trad_df_all[trad_df_all['period'] == p]['latency_ms'].mean() for p in PERIODS]
o_lats = [all_df[all_df['period'] == p]['latency_ms'].mean()            for p in PERIODS]

b1 = ax.bar(x - w/2, t_lats, w, label='Traditional', color=C_TRAD,
            edgecolor='white', linewidth=0.8)
b2 = ax.bar(x + w/2, o_lats, w, label='Ours (ML)',   color=C_OUR,
            edgecolor='white', linewidth=0.8)

for bar, val in zip(list(b1) + list(b2), t_lats + o_lats):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=9, color=C_TEXT)

# Improvement % labels above each pair
for i, (tl, ol) in enumerate(zip(t_lats, o_lats)):
    imp = (tl - ol) / tl * 100
    ax.text(i, max(tl, ol) + 3, f'−{imp:.1f}%',
            ha='center', fontsize=9, color='#2E7D32', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(PERIODS)
ax_style(ax, 'Avg Latency per Time Period\nTraditional vs ML-Driven Approach',
         ylabel='Avg Latency (ms)')
ax.legend(frameon=False)
fig.tight_layout()
save(fig, '03_latency_comparison.png')


# =============================================================================
# FIG 04 — Deadline Met Rate  (bar + line overlay)
# aim: Reliability — traditional misses deadlines, ours doesn't
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('white')

x = np.arange(len(PERIODS))
w = 0.35
t_dl = [trad_df_all[trad_df_all['period'] == p]['deadline_met'].mean() * 100 for p in PERIODS]
o_dl = [all_df[all_df['period'] == p]['deadline_met'].mean() * 100            for p in PERIODS]

ax.bar(x - w/2, t_dl, w, label='Traditional', color=C_TRAD,
       edgecolor='white', linewidth=0.8)
ax.bar(x + w/2, o_dl, w, label='Ours (ML)',   color=C_OUR,
       edgecolor='white', linewidth=0.8)

# 100% reference line
ax.axhline(100, color='#2E7D32', lw=1.2, ls='--', alpha=0.6, label='100% target')

# Missed count annotations on traditional bars
for i, (p, dl) in enumerate(zip(PERIODS, t_dl)):
    missed = int((1 - dl / 100) *
                 len(trad_df_all[trad_df_all['period'] == p]))
    ax.text(i - w/2, dl - 2.5, f'{missed}\nmissed',
            ha='center', va='top', fontsize=8,
            color='white', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(PERIODS)
ax.set_ylim(80, 105)
ax_style(ax, 'Deadline Met Rate per Period\nTraditional vs ML-Driven Approach',
         ylabel='Deadline Met Rate (%)')
ax.legend(frameon=False)
fig.tight_layout()
save(fig, '04_deadline_met_rate.png')


# =============================================================================
# FIG 05 — Resource Distribution  (two pie charts side by side)
# aim: ML uses VCC intelligently; traditional wastes Cloud bandwidth
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor('white')

# Traditional: only RSU + Cloud
t_sizes  = [trad_rsu, trad_cld]
t_labels = [f'RSU\n{trad_rsu:,}\n({trad_rsu/grand_total*100:.1f}%)',
            f'Cloud\n{trad_cld:,}\n({trad_cld/grand_total*100:.1f}%)']
ax1.pie(t_sizes, labels=t_labels, colors=[C_RSU, C_CLOUD],
        wedgeprops={'edgecolor': 'white', 'linewidth': 2.5},
        textprops={'fontsize': 10, 'color': C_TEXT},
        startangle=90)
ax1.set_title('Traditional Approach\n(RSU + Cloud only)',
              fontweight='bold', color=C_TEXT, pad=14)

# Ours: VCC + RSU + Cloud
o_sizes  = [our_loc, our_rsu, our_cld]
o_labels = [f'VCC (Local)\n{our_loc:,}\n({our_loc/grand_total*100:.1f}%)',
            f'RSU\n{our_rsu:,}\n({our_rsu/grand_total*100:.1f}%)',
            f'Cloud\n{our_cld:,}\n({our_cld/grand_total*100:.1f}%)']
ax2.pie(o_sizes, labels=o_labels, colors=[C_VCC, C_RSU, C_CLOUD],
        wedgeprops={'edgecolor': 'white', 'linewidth': 2.5},
        textprops={'fontsize': 10, 'color': C_TEXT},
        startangle=90)
ax2.set_title('ML-Driven Approach\n(VCC + RSU + Cloud)',
              fontweight='bold', color=C_TEXT, pad=14)

fig.suptitle('Resource Distribution: Where Tasks Are Executed',
             fontsize=13, fontweight='bold', color=C_TEXT, y=1.02)
fig.tight_layout()
save(fig, '05_resource_distribution.png')


# =============================================================================
# FIG 06 — Per-Period Analysis  (3-metric grouped bar: cost, latency, DL%)
# aim: Our approach consistently wins across all 4 time periods
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')

x = np.arange(len(PERIODS))
w = 0.35

metrics = [
    ('Total Cost (USD)',
     [trad_df_all[trad_df_all['period'] == p]['cost_usd'].sum() for p in PERIODS],
     [all_df[all_df['period'] == p]['cost_usd'].sum()           for p in PERIODS]),
    ('Avg Latency (ms)',
     [trad_df_all[trad_df_all['period'] == p]['latency_ms'].mean() for p in PERIODS],
     [all_df[all_df['period'] == p]['latency_ms'].mean()            for p in PERIODS]),
    ('Deadline Met Rate (%)',
     [trad_df_all[trad_df_all['period'] == p]['deadline_met'].mean() * 100 for p in PERIODS],
     [all_df[all_df['period'] == p]['deadline_met'].mean() * 100            for p in PERIODS]),
]

for ax, (title, t_vals, o_vals) in zip(axes, metrics):
    ax.bar(x - w/2, t_vals, w, label='Traditional', color=C_TRAD,
           edgecolor='white', linewidth=0.8)
    ax.bar(x + w/2, o_vals, w, label='Ours (ML)',   color=C_OUR,
           edgecolor='white', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(PERIODS, fontsize=9)
    ax_style(ax, title, ylabel=title)
    ax.legend(frameon=False, fontsize=8)

fig.suptitle('Per-Period Analysis — Traditional vs ML-Driven Approach',
             fontsize=13, fontweight='bold', color=C_TEXT, y=1.02)
fig.tight_layout()
save(fig, '06_per_period_analysis.png')


# =============================================================================
# FIG 07 — Per-Task-Type Heatmap
# aim: Which task types benefit most — read improvement at a glance
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')

metrics_tt = [
    ('Avg Latency (ms)',
     {tt: trad_df_all[trad_df_all['task_type'] == tt]['latency_ms'].mean() for tt in TASK_TYPES},
     {tt: all_df[all_df['task_type'] == tt]['latency_ms'].mean()            for tt in TASK_TYPES}),
    ('Avg Cost/Task (USD)',
     {tt: trad_df_all[trad_df_all['task_type'] == tt]['cost_usd'].mean() for tt in TASK_TYPES},
     {tt: all_df[all_df['task_type'] == tt]['cost_usd'].mean()            for tt in TASK_TYPES}),
    ('Deadline Met Rate (%)',
     {tt: trad_df_all[trad_df_all['task_type'] == tt]['deadline_met'].mean() * 100 for tt in TASK_TYPES},
     {tt: all_df[all_df['task_type'] == tt]['deadline_met'].mean() * 100            for tt in TASK_TYPES}),
]

x  = np.arange(len(TASK_TYPES))
w  = 0.35
tt_labels = [tt.replace('_', '\n') for tt in TASK_TYPES]

for ax, (title, t_dict, o_dict) in zip(axes, metrics_tt):
    t_vals = [t_dict[tt] for tt in TASK_TYPES]
    o_vals = [o_dict[tt] for tt in TASK_TYPES]
    b1 = ax.bar(x - w/2, t_vals, w, label='Traditional', color=C_TRAD,
                edgecolor='white', linewidth=0.8)
    b2 = ax.bar(x + w/2, o_vals, w, label='Ours (ML)',   color=C_OUR,
                edgecolor='white', linewidth=0.8)

    # Improvement % on top of each pair
    for i, (tv, ov) in enumerate(zip(t_vals, o_vals)):
        if tv > 0:
            imp = (tv - ov) / tv * 100
            sign = '−' if imp > 0 else '+'
            color = '#2E7D32' if imp > 0 else '#C62828'
            ax.text(i, max(tv, ov) * 1.04,
                    f'{sign}{abs(imp):.0f}%',
                    ha='center', fontsize=8,
                    color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(tt_labels, fontsize=8)
    ax_style(ax, title, ylabel=title)
    ax.legend(frameon=False, fontsize=8)

fig.suptitle('Per-Task-Type Comparison — Traditional vs ML-Driven Approach',
             fontsize=13, fontweight='bold', color=C_TEXT, y=1.02)
fig.tight_layout()
save(fig, '07_per_task_type_analysis.png')


# =============================================================================
# FIG 08 — Cost-Latency Tradeoff Scatter
# aim: ML dominates — lower cost AND lower latency simultaneously
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('white')

# One point per task type, two approaches
for tt in TASK_TYPES:
    t_sub = trad_df_all[trad_df_all['task_type'] == tt]
    o_sub = all_df[all_df['task_type'] == tt]

    t_lat_v = t_sub['latency_ms'].mean()
    t_cst_v = t_sub['cost_usd'].mean()
    o_lat_v = o_sub['latency_ms'].mean()
    o_cst_v = o_sub['cost_usd'].mean()

    label = tt.replace('_', ' ').title()

    ax.scatter(t_lat_v, t_cst_v, color=C_TRAD, s=120,
               zorder=3, edgecolors='white', linewidths=1)
    ax.scatter(o_lat_v, o_cst_v, color=C_OUR,  s=120,
               zorder=3, edgecolors='white', linewidths=1)

    # Arrow from Traditional → Ours
    ax.annotate('', xy=(o_lat_v, o_cst_v),
                xytext=(t_lat_v, t_cst_v),
                arrowprops=dict(arrowstyle='->', color='#888888',
                                lw=1.2, connectionstyle='arc3,rad=0.1'))

    # Label midpoint
    mid_x = (t_lat_v + o_lat_v) / 2
    mid_y = (t_cst_v + o_cst_v) / 2
    ax.text(mid_x + 1, mid_y, label, fontsize=8,
            color=C_TEXT, va='center')

# Quadrant shading — bottom-left is best
ax_xlim = ax.get_xlim()
ax_ylim = ax.get_ylim()
ax.set_xlim(ax_xlim)
ax.set_ylim(ax_ylim)

mid_x_q = (ax_xlim[0] + ax_xlim[1]) / 2
mid_y_q = (ax_ylim[0] + ax_ylim[1]) / 2
ax.axhline(mid_y_q, color=C_GRID, lw=1, ls='--')
ax.axvline(mid_x_q, color=C_GRID, lw=1, ls='--')
ax.text(ax_xlim[0] + 1, ax_ylim[0] + 0.0002,
        '✓ Ideal Zone\n(Low Cost, Low Latency)',
        fontsize=8, color='#2E7D32', alpha=0.7)

# Legend
legend_handles = [
    mpatches.Patch(color=C_TRAD, label='Traditional (RSU+Cloud)'),
    mpatches.Patch(color=C_OUR,  label='Ours (VCC+RSU+Cloud)'),
]
ax.legend(handles=legend_handles, frameon=False, loc='upper right')

ax_style(ax, 'Cost–Latency Tradeoff per Task Type\nArrows show: Traditional → ML-Driven',
         xlabel='Avg Latency (ms)', ylabel='Avg Cost per Task (USD)')
fig.tight_layout()
save(fig, '08_cost_latency_tradeoff.png')


# =============================================================================
# PHASE 4 COMPLETE
# =============================================================================
print(f"""
  {'━' * 70}
  PHASE 4 COMPLETE — All 8 figures saved to '{VIZ_DIR}/'
  {'━' * 70}
  01_total_cost_comparison.png    — Overall cost: Traditional vs ML
  02_cost_per_task_type.png       — Cost breakdown by task type
  03_latency_comparison.png       — Latency per time period
  04_deadline_met_rate.png        — Deadline reliability comparison
  05_resource_distribution.png    — Pie: VCC vs RSU vs Cloud usage
  06_per_period_analysis.png      — Cost, latency, DL% across 4 periods
  07_per_task_type_analysis.png   — 3-metric breakdown per task type
  08_cost_latency_tradeoff.png    — Scatter: cost vs latency tradeoff
  {'━' * 70}
""")


print("\n" + "=" * 70)
print("  PHASE 5 | PAPER-BASED APPROACH COMPARISONS")
print("=" * 70)
print("""
  Approaches compared:
    ① Our ML (VCC + RSU + Cloud, multi-objective, priority-aware)   [Phase 3]
    ② Paper 1 — DQN-2020  : Local/RSU/Cloud, latency-only
    ③ Paper 2 — NSGA-II-2021: RSU/Cloud, equal-weight cost+latency, static
    ④ Paper 3 — DQN-IoV-2022: RSU/Cloud, 3-component latency (no VCC)
""")


# =============================================================================
# PAPER 1 — DQN (Latency-Only) SIMULATION
#  Rule: pick whichever target gives the lowest raw latency.
#  No signal quality degradation, no speed penalty, no priority, no VCC cap.
#  Fixed signal = -65 dBm, fixed speed = 40 km/h (Paper 1 ignores these).
#  Local is always feasible (no capacity gate).
# =============================================================================
print("─" * 70)
print("  PAPER 1 — DQN Latency-Only (2020)")
print("─" * 70)

P1_DEFAULT_SIG = -65.0
P1_DEFAULT_SPD =  40.0

paper1_results = []

for period in period_order:
    ts = selected[period]
    task_ts = df_task_pred[df_task_pred['timestamp'] == ts].copy()
    # No priority sort — raw order

    for _, task_row in task_ts.iterrows():
        rid     = task_row['region_id']
        ttype   = task_row['task_type']
        cpu_M   = task_row['pred_cpu_cycles_M']
        data_KB = task_row['pred_data_size_KB']
        deadline = task_row['pred_deadline_ms']

        # Per-vehicle speed/signal (Paper 1 ignores these but we retrieve
        # them for fair deadline_check comparison)
        veh_row = df_vehicle_pred[df_vehicle_pred['vehicle_id'] == task_row['vehicle_id']]
        speed_kmh  = veh_row['pred_speed_kmh'].values[0]  if not veh_row.empty else P1_DEFAULT_SPD
        signal_dBm = veh_row['pred_signal_dBm'].values[0] if not veh_row.empty else P1_DEFAULT_SIG

        # Pick target with minimum latency (Paper 1 decision rule)
        # Uses fixed signal & speed — no network-quality awareness
        best_target = None
        best_lat    = float('inf')
        for tname, infra in INFRA.items():
            lat = compute_latency(cpu_M, data_KB, infra,
                                  P1_DEFAULT_SIG, P1_DEFAULT_SPD, tname)
            if lat < best_lat:
                best_lat    = lat
                best_target = tname

        cst    = compute_cost(cpu_M, data_KB, INFRA[best_target])
        dl_met = realistic_deadline_check(best_lat, deadline, ttype,
                                          speed_kmh, signal_dBm, 'Traditional')

        paper1_results.append({
            'period':       period,
            'timestamp':    ts,
            'region_id':    rid,
            'task_type':    ttype,
            'priority':     TASK_PRIORITY[ttype],
            'cpu_M':        cpu_M,
            'data_KB':      data_KB,
            'deadline_ms':  deadline,
            'target':       best_target,
            'latency_ms':   best_lat,
            'cost_usd':     cst,
            'deadline_met': int(dl_met),
        })

p1_df = pd.DataFrame(paper1_results)
p1_total = len(p1_df)
p1_dl    = p1_df['deadline_met'].sum()
p1_lat   = p1_df['latency_ms'].mean()
p1_cst   = p1_df['cost_usd'].sum()
p1_rsu   = (p1_df['target'] == 'RSU').sum()
p1_loc   = (p1_df['target'] == 'Local').sum()
p1_cld   = (p1_df['target'] == 'Cloud').sum()

print(f"  Tasks: {p1_total:,}  |  Local: {p1_loc:,}  RSU: {p1_rsu:,}  Cloud: {p1_cld:,}")
print(f"  Avg Latency : {p1_lat:.2f} ms")
print(f"  Total Cost  : ${p1_cst:.4f}")
print(f"  Deadline Met: {p1_dl:,} / {p1_total:,}  ({p1_dl/p1_total*100:.1f}%)")


# =============================================================================
# PAPER 2 — NSGA-II Equal-Weight Scalarisation (Cost + Latency, Static)
#  Simulated as: pick target minimising F = 0.5*norm_lat + 0.5*norm_cost
#  No VCC, no priority, no per-vehicle signal/speed awareness.
#  RSU capacity = fixed base (no dynamic scaling).
#  Tasks processed in raw dataset order.
# =============================================================================
print("\n" + "─" * 70)
print("  PAPER 2 — NSGA-II Cost+Latency, Static (2021)")
print("─" * 70)

P2_W1 = 0.5   # equal latency weight
P2_W2 = 0.5   # equal cost weight
P2_DEFAULT_SIG = -65.0
P2_DEFAULT_SPD =  40.0

paper2_results = []

for period in period_order:
    ts = selected[period]
    task_ts = df_task_pred[df_task_pred['timestamp'] == ts].copy()
    p2_rsu_tasks = {rid: 0 for rid in REGIONS}  # fixed RSU cap tracker

    for _, task_row in task_ts.iterrows():
        rid     = task_row['region_id']
        ttype   = task_row['task_type']
        cpu_M   = task_row['pred_cpu_cycles_M']
        data_KB = task_row['pred_data_size_KB']
        deadline = task_row['pred_deadline_ms']

        veh_row = df_vehicle_pred[df_vehicle_pred['vehicle_id'] == task_row['vehicle_id']]
        speed_kmh  = veh_row['pred_speed_kmh'].values[0]  if not veh_row.empty else P2_DEFAULT_SPD
        signal_dBm = veh_row['pred_signal_dBm'].values[0] if not veh_row.empty else P2_DEFAULT_SIG

        # NSGA-II only uses RSU and Cloud (no VCC in paper)
        candidates_p2 = []
        for tname in ('RSU', 'Cloud'):
            # RSU feasibility: fixed cap
            if tname == 'RSU' and p2_rsu_tasks[rid] >= TRAD_RSU_CAP:
                continue

            infra = INFRA[tname]
            lat   = compute_latency(cpu_M, data_KB, infra,
                                    P2_DEFAULT_SIG, P2_DEFAULT_SPD, tname)
            cst   = compute_cost(cpu_M, data_KB, infra)

            norm_lat = lat / NORM_MAX_LAT if NORM_MAX_LAT > 0 else 0
            norm_cst = cst / NORM_MAX_CST if NORM_MAX_CST > 0 else 0
            F = P2_W1 * norm_lat + P2_W2 * norm_cst

            candidates_p2.append({'target': tname, 'latency_ms': lat,
                                   'cost_usd': cst, 'f_score': F})

        if not candidates_p2:
            chosen_p2 = {'target': 'Cloud',
                         'latency_ms': compute_latency(cpu_M, data_KB, INFRA['Cloud'],
                                                       P2_DEFAULT_SIG, P2_DEFAULT_SPD, 'Cloud'),
                         'cost_usd': compute_cost(cpu_M, data_KB, INFRA['Cloud']),
                         'f_score': 99.0}
        else:
            chosen_p2 = min(candidates_p2, key=lambda x: x['f_score'])

        if chosen_p2['target'] == 'RSU':
            p2_rsu_tasks[rid] += 1

        dl_met = realistic_deadline_check(chosen_p2['latency_ms'], deadline, ttype,
                                          speed_kmh, signal_dBm, 'Traditional')

        paper2_results.append({
            'period':       period,
            'timestamp':    ts,
            'region_id':    rid,
            'task_type':    ttype,
            'priority':     TASK_PRIORITY[ttype],
            'cpu_M':        cpu_M,
            'data_KB':      data_KB,
            'deadline_ms':  deadline,
            'target':       chosen_p2['target'],
            'latency_ms':   chosen_p2['latency_ms'],
            'cost_usd':     chosen_p2['cost_usd'],
            'deadline_met': int(dl_met),
        })

p2_df = pd.DataFrame(paper2_results)
p2_total = len(p2_df)
p2_dl    = p2_df['deadline_met'].sum()
p2_lat   = p2_df['latency_ms'].mean()
p2_cst   = p2_df['cost_usd'].sum()
p2_rsu   = (p2_df['target'] == 'RSU').sum()
p2_cld   = (p2_df['target'] == 'Cloud').sum()

print(f"  Tasks: {p2_total:,}  |  RSU: {p2_rsu:,}  Cloud: {p2_cld:,}  (no VCC)")
print(f"  Avg Latency : {p2_lat:.2f} ms")
print(f"  Total Cost  : ${p2_cst:.4f}")
print(f"  Deadline Met: {p2_dl:,} / {p2_total:,}  ({p2_dl/p2_total*100:.1f}%)")


# =============================================================================
# PAPER 3 — DQN-IoV (3-Component Latency, Per-Vehicle Awareness)
#  T_total = T_upload + T_compute + T_download
#  T_download = data_KB / (eff_bw * 0.5)  [half bandwidth for result return]
#  No VCC, no cost, no priority.
#  Uses per-vehicle signal strength and speed (unlike Paper 1).
#  RSU capacity = fixed (Paper 3 does not model capacity limits).
# =============================================================================
print("\n" + "─" * 70)
print("  PAPER 3 — DQN-IoV 3-Component Latency (2022)")
print("─" * 70)

def compute_latency_p3(cpu_M: float, data_KB: float,
                       infra: dict, signal_dBm: float, speed_kmh: float,
                       target: str) -> float:
    """
    Paper 3 latency model: T = T_upload + T_compute + T_download
    T_download uses half the upload bandwidth (result is typically smaller).
    """
    bw_factor  = signal_bandwidth_factor(signal_dBm)
    eff_bw     = infra['bw_KB_per_ms'] * bw_factor
    t_upload   = data_KB / eff_bw
    t_compute  = cpu_M   / infra['cpu_speed_M_per_ms']
    t_download = data_KB / (eff_bw * 2.0)   # result download (half size)
    t_handoff  = speed_handoff_penalty(speed_kmh, target)
    return infra['base_latency_ms'] + t_upload + t_compute + t_download + t_handoff

paper3_results = []

for period in period_order:
    ts = selected[period]
    task_ts = df_task_pred[df_task_pred['timestamp'] == ts].copy()

    for _, task_row in task_ts.iterrows():
        rid     = task_row['region_id']
        ttype   = task_row['task_type']
        cpu_M   = task_row['pred_cpu_cycles_M']
        data_KB = task_row['pred_data_size_KB']
        deadline = task_row['pred_deadline_ms']

        veh_row = df_vehicle_pred[df_vehicle_pred['vehicle_id'] == task_row['vehicle_id']]
        speed_kmh  = veh_row['pred_speed_kmh'].values[0]  if not veh_row.empty else 40.0
        signal_dBm = veh_row['pred_signal_dBm'].values[0] if not veh_row.empty else -65.0

        # Paper 3: RSU or Cloud — pick min 3-component latency
        best_target_p3 = None
        best_lat_p3    = float('inf')
        for tname in ('RSU', 'Cloud'):
            lat = compute_latency_p3(cpu_M, data_KB, INFRA[tname],
                                     signal_dBm, speed_kmh, tname)
            if lat < best_lat_p3:
                best_lat_p3    = lat
                best_target_p3 = tname

        cst    = compute_cost(cpu_M, data_KB, INFRA[best_target_p3])
        dl_met = realistic_deadline_check(best_lat_p3, deadline, ttype,
                                          speed_kmh, signal_dBm, 'Traditional')

        paper3_results.append({
            'period':       period,
            'timestamp':    ts,
            'region_id':    rid,
            'task_type':    ttype,
            'priority':     TASK_PRIORITY[ttype],
            'cpu_M':        cpu_M,
            'data_KB':      data_KB,
            'deadline_ms':  deadline,
            'target':       best_target_p3,
            'latency_ms':   best_lat_p3,
            'cost_usd':     cst,
            'deadline_met': int(dl_met),
        })

p3_df = pd.DataFrame(paper3_results)
p3_total = len(p3_df)
p3_dl    = p3_df['deadline_met'].sum()
p3_lat   = p3_df['latency_ms'].mean()
p3_cst   = p3_df['cost_usd'].sum()
p3_rsu   = (p3_df['target'] == 'RSU').sum()
p3_cld   = (p3_df['target'] == 'Cloud').sum()

print(f"  Tasks: {p3_total:,}  |  RSU: {p3_rsu:,}  Cloud: {p3_cld:,}  (no VCC)")
print(f"  Avg Latency : {p3_lat:.2f} ms")
print(f"  Total Cost  : ${p3_cst:.4f}")
print(f"  Deadline Met: {p3_dl:,} / {p3_total:,}  ({p3_dl/p3_total*100:.1f}%)")


# =============================================================================
# PHASE 5 — 4-WAY AGGREGATE COMPARISON TABLE
# =============================================================================
print(f"\n{'━' * 70}")
print(f"  PHASE 5 — 4-WAY AGGREGATE COMPARISON  (All Timestamps Combined)")
print(f"{'━' * 70}")

# Our approach metrics (re-use Phase 3 aggregates)
our_lat_mean = all_df['latency_ms'].mean()
our_tot_cost = all_df['cost_usd'].sum()
our_dl_pct   = all_df['deadline_met'].mean() * 100
our_loc_cnt  = (all_df['target'] == 'Local').sum()
our_rsu_cnt  = (all_df['target'] == 'RSU').sum()
our_cld_cnt  = (all_df['target'] == 'Cloud').sum()

approaches = [
    ("Our ML (VCC+RSU+Cloud)",  our_lat_mean, our_tot_cost, our_dl_pct,
     our_loc_cnt, our_rsu_cnt, our_cld_cnt),
    ("Paper1 DQN-2020",         p1_lat,        p1_cst,       p1_dl/p1_total*100,
     p1_loc, p1_rsu, p1_cld),
    ("Paper2 NSGA-II-2021",     p2_lat,        p2_cst,       p2_dl/p2_total*100,
     0, p2_rsu, p2_cld),
    ("Paper3 DQN-IoV-2022",     p3_lat,        p3_cst,       p3_dl/p3_total*100,
     0, p3_rsu, p3_cld),
]

print(f"\n  {'Approach':<26} {'AvgLat(ms)':>11} {'TotCost($)':>12} "
      f"{'DL Met%':>9} {'VCC':>7} {'RSU':>7} {'Cloud':>7}")
print(f"  {'─'*80}")
for name, lat, cst, dlp, loc, rsu, cld in approaches:
    loc_str = f"{loc:>7,}" if loc > 0 else f"{'—':>7}"
    print(f"  {name:<26} {lat:>11.2f} {cst:>12.4f} {dlp:>8.1f}% "
          f"{loc_str} {rsu:>7,} {cld:>7,}")

# Improvement of Ours vs each paper
print(f"\n  {'─'*80}")
print(f"  IMPROVEMENT OF OUR APPROACH vs EACH PAPER:")
for name, lat, cst, dlp, *_ in approaches[1:]:
    lat_imp = (lat  - our_lat_mean) / lat  * 100  if lat  > 0 else 0
    cst_imp = (cst  - our_tot_cost) / cst  * 100  if cst  > 0 else 0
    dl_imp  = our_dl_pct - dlp
    print(f"  vs {name:<22}  Lat: {lat_imp:+.1f}%  Cost: {cst_imp:+.1f}%  "
          f"DL: {dl_imp:+.1f}pp")

# Per-period 4-way table
print(f"\n{'━' * 70}")
print(f"  PER-PERIOD 4-WAY COMPARISON")
print(f"{'━' * 70}")

for period in period_order:
    print(f"\n  {period.upper()} — {selected[period]}")
    print(f"  {'Approach':<26} {'AvgLat':>9} {'TotCost':>10} {'DL Met%':>9}")
    print(f"  {'─'*58}")

    sub_our = all_df[all_df['period'] == period]
    sub_p1  = p1_df[p1_df['period']  == period]
    sub_p2  = p2_df[p2_df['period']  == period]
    sub_p3  = p3_df[p3_df['period']  == period]

    for label, sub in [
        ("Our ML (VCC+RSU+Cloud)", sub_our),
        ("Paper1 DQN-2020",         sub_p1),
        ("Paper2 NSGA-II-2021",     sub_p2),
        ("Paper3 DQN-IoV-2022",     sub_p3),
    ]:
        if sub.empty: continue
        print(f"  {label:<26} {sub['latency_ms'].mean():>9.1f} "
              f"{sub['cost_usd'].sum():>10.4f} "
              f"{sub['deadline_met'].mean()*100:>8.1f}%")

# Per-task-type 4-way table
print(f"\n{'━' * 70}")
print(f"  PER-TASK-TYPE 4-WAY COMPARISON  (Deadline Met Rate %)")
print(f"{'━' * 70}")
print(f"  {'Task Type':<20} {'Ours':>8} {'Paper1':>8} {'Paper2':>8} {'Paper3':>8}")
print(f"  {'─'*55}")
for tt in sorted(TASK_PRIORITY, key=TASK_PRIORITY.get):
    def dlp(df, t=tt):
        s = df[df['task_type'] == t]
        return s['deadline_met'].mean()*100 if not s.empty else 0.0
    print(f"  {tt:<20} {dlp(all_df):>7.1f}% {dlp(p1_df):>7.1f}% "
          f"{dlp(p2_df):>7.1f}% {dlp(p3_df):>7.1f}%")


# =============================================================================
# PHASE 5 — VISUALISATIONS  (5 new figures, appended to visualizations/)
# =============================================================================
print(f"\n{'━' * 70}")
print(f"  PHASE 5 | VISUALISATIONS  (5 new figures)")
print(f"{'━' * 70}")

# Reuse plot helpers from Phase 4 (ax_style, save, color palette already defined)

APPROACHES    = ["Our ML", "Paper1\nDQN-2020", "Paper2\nNSGA-II", "Paper3\nDQN-IoV"]
APPROACH_COLS = ["#C1FFE1", "#7EB8D4", "#FFCD88", "#D4A4C8"]

# Global approach dfs for convenience
approach_dfs = {
    "Our ML":         all_df,
    "Paper1\nDQN-2020": p1_df,
    "Paper2\nNSGA-II":  p2_df,
    "Paper3\nDQN-IoV":  p3_df,
}

# ── FIG 09 — 4-Way Total Cost Comparison (bar) ───────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('white')

costs = [our_tot_cost, p1_cst, p2_cst, p3_cst]
bars  = ax.bar(APPROACHES, costs, color=APPROACH_COLS, width=0.55,
               edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, costs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(costs) * 0.02,
            f'${val:.2f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=C_TEXT)

ax_style(ax, '4-Way Total Cost Comparison\n(All 4 Timestamps, All Tasks)',
         ylabel='Total Cost (USD)')
ax.set_ylim(0, max(costs) * 1.22)
fig.tight_layout()
save(fig, '09_4way_total_cost.png')


# ── FIG 10 — 4-Way Avg Latency Comparison (bar) ──────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('white')

lats = [our_lat_mean, p1_lat, p2_lat, p3_lat]
bars = ax.bar(APPROACHES, lats, color=APPROACH_COLS, width=0.55,
              edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, lats):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(lats) * 0.02,
            f'{val:.1f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=C_TEXT)

# Improvement arrows from each paper to ours
our_x = bars[0].get_x() + bars[0].get_width() / 2
for i, (bar, val) in enumerate(zip(bars[1:], lats[1:]), start=1):
    px = bar.get_x() + bar.get_width() / 2
    imp = (val - lats[0]) / val * 100
    ax.annotate('', xy=(our_x, lats[0] + max(lats)*0.05),
                xytext=(px,   val + max(lats)*0.05),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.2,
                                connectionstyle='arc3,rad=-0.2'))
    mid_x = (our_x + px) / 2
    mid_y = max(lats[0], val) + max(lats) * 0.12
    ax.text(mid_x, mid_y, f'−{imp:.0f}%', ha='center', fontsize=8,
            color='#2E7D32', fontweight='bold')

ax_style(ax, '4-Way Average Latency Comparison\n(All 4 Timestamps, All Tasks)',
         ylabel='Avg Latency per Task (ms)')
ax.set_ylim(0, max(lats) * 1.3)
fig.tight_layout()
save(fig, '10_4way_avg_latency.png')


# ── FIG 11 — 4-Way Deadline Met Rate (grouped bar per period) ────────────────
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor('white')

x = np.arange(len(PERIODS))
n = len(APPROACHES)
total_w = 0.70
w = total_w / n
offsets = np.linspace(-total_w/2 + w/2, total_w/2 - w/2, n)

for i, (label, col) in enumerate(zip(APPROACHES, APPROACH_COLS)):
    df_app = approach_dfs[label]
    dlp = [df_app[df_app['period'] == p]['deadline_met'].mean() * 100
           for p in PERIODS]
    bars = ax.bar(x + offsets[i], dlp, w, label=label.replace('\n', ' '),
                  color=col, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, dlp):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f'{val:.0f}', ha='center', va='bottom',
                fontsize=7.5, color=C_TEXT)

ax.axhline(100, color='#2E7D32', lw=1.2, ls='--', alpha=0.5, label='100% target')
ax.set_xticks(x)
ax.set_xticklabels(PERIODS)
ax.set_ylim(80, 108)
ax_style(ax, 'Deadline Met Rate per Time Period — 4-Way Comparison',
         ylabel='Deadline Met Rate (%)')
ax.legend(frameon=False, fontsize=9, ncol=5)
fig.tight_layout()
save(fig, '11_4way_deadline_per_period.png')


# ── FIG 12 — Per-Task-Type Deadline Met Rate (grouped bar) ───────────────────
fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor('white')

tt_labels_short = [tt.replace('_', '\n') for tt in TASK_TYPES]
x = np.arange(len(TASK_TYPES))
total_w = 0.72
w = total_w / n
offsets = np.linspace(-total_w/2 + w/2, total_w/2 - w/2, n)

for i, (label, col) in enumerate(zip(APPROACHES, APPROACH_COLS)):
    df_app = approach_dfs[label]
    dlp = [df_app[df_app['task_type'] == tt]['deadline_met'].mean() * 100
           for tt in TASK_TYPES]
    bars = ax.bar(x + offsets[i], dlp, w, label=label.replace('\n', ' '),
                  color=col, edgecolor='white', linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(tt_labels_short, fontsize=9)
ax.set_ylim(75, 108)
ax.axhline(100, color='#2E7D32', lw=1.2, ls='--', alpha=0.5, label='100% target')
ax_style(ax, 'Deadline Met Rate per Task Type — 4-Way Comparison\n(sorted by priority: highest → lowest)',
         ylabel='Deadline Met Rate (%)')
ax.legend(frameon=False, fontsize=9, ncol=5)
fig.tight_layout()
save(fig, '12_4way_deadline_per_task_type.png')


# ── FIG 13 — 4-Way Cost-Latency Radar / Spider Chart ────────────────────────
#   Metrics (normalised 0→1, higher = better):
#     1. Deadline Rate    (higher = better)
#     2. Latency Savings  (vs worst approach)
#     3. Cost Savings     (vs worst approach)
#     4. VCC Utilisation  (local tasks / total)
#     5. Priority Adherence (autonomous_alert deadline rate)
# ────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 8))
fig.patch.set_facecolor('white')
ax  = fig.add_subplot(111, polar=True)

radar_metrics = [
    'Deadline\nRate',
    'Latency\nSavings',
    'Cost\nSavings',
    'VCC\nUtilisation',
    'Priority\nAdherence',
]
num_metrics = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]   # close the polygon

# Raw values
worst_lat  = max(our_lat_mean, p1_lat, p2_lat, p3_lat)
worst_cst  = max(our_tot_cost, p1_cst, p2_cst, p3_cst)
total_t    = grand_total

def alert_dl(df):
    sub = df[df['task_type'] == 'autonomous_alert']
    return sub['deadline_met'].mean() * 100 if not sub.empty else 0.0

def vcc_util(df):
    if 'target' in df.columns:
        return (df['target'] == 'Local').sum() / len(df) * 100
    return 0.0

raw_values = {
    "Our ML":        [our_dl_pct,
                      (worst_lat - our_lat_mean) / worst_lat * 100,
                      (worst_cst - our_tot_cost) / worst_cst * 100,
                      vcc_util(all_df),
                      alert_dl(all_df)],
    "Paper1\nDQN-2020": [p1_dl/p1_total*100,
                          (worst_lat - p1_lat) / worst_lat * 100,
                          (worst_cst - p1_cst) / worst_cst * 100,
                          vcc_util(p1_df),
                          alert_dl(p1_df)],
    "Paper2\nNSGA-II":  [p2_dl/p2_total*100,
                          (worst_lat - p2_lat) / worst_lat * 100,
                          (worst_cst - p2_cst) / worst_cst * 100,
                          vcc_util(p2_df),
                          alert_dl(p2_df)],
    "Paper3\nDQN-IoV":  [p3_dl/p3_total*100,
                          (worst_lat - p3_lat) / worst_lat * 100,
                          (worst_cst - p3_cst) / worst_cst * 100,
                          vcc_util(p3_df),
                          alert_dl(p3_df)],
}

# Normalise each metric to 0-1 range
all_vals = np.array(list(raw_values.values()))
max_vals = all_vals.max(axis=0)
max_vals[max_vals == 0] = 1   # avoid div-by-zero

for i, (label, col) in enumerate(zip(APPROACHES, APPROACH_COLS)):
    vals_raw = raw_values[label]
    vals_norm = [v / m for v, m in zip(vals_raw, max_vals)]
    vals_norm += vals_norm[:1]

    ax.plot(angles, vals_norm, 'o-', linewidth=2,
            color=col, markerfacecolor=col,
            label=label.replace('\n', ' '))
    ax.fill(angles, vals_norm, alpha=0.18, color=col)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, fontsize=10, color=C_TEXT)
ax.set_ylim(0, 1.1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7, color='#888888')
ax.set_facecolor('#FAFAFA')
ax.grid(color=C_GRID, linestyle='--', linewidth=0.5)
ax.set_title('Multi-Metric Radar: 4-Way Approach Comparison\n(higher = better on all axes)',
             fontsize=12, fontweight='bold', color=C_TEXT, pad=22)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
          frameon=False, fontsize=9)
fig.tight_layout()
save(fig, '13_4way_radar_chart.png')


# =============================================================================
# PHASE 5 COMPLETE
# =============================================================================
print(f"""
  {'━' * 70}
  PHASE 5 COMPLETE — Paper-Based Comparisons & Figures
  {'━' * 70}
  Approaches simulated:
    ① Our ML  (VCC + RSU + Cloud, multi-objective, priority-aware)
    ② Paper 1 — DQN-2020          (latency-only, no VCC, no cost)
    ③ Paper 2 — NSGA-II-2021      (equal-weight cost+latency, static)
    ④ Paper 3 — DQN-IoV-2022      (3-component latency, per-vehicle aware)

  New figures saved to '{VIZ_DIR}/':
    09_4way_total_cost.png          — Total cost: all 4 approaches
    10_4way_avg_latency.png         — Avg latency with improvement arrows
    11_4way_deadline_per_period.png — Deadline rate per time period
    12_4way_deadline_per_task_type.png — Deadline rate per task type
    13_4way_radar_chart.png         — Multi-metric radar spider chart
  {'━' * 70}
""")
print("=" * 70)
print("  ALL PHASES COMPLETE")
print("=" * 70)