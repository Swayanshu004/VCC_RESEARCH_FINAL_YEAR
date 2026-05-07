"""
=====================================================================
  SIMULATION: ML-DRIVEN MULTI-OBJECTIVE COST & LATENCY OPTIMIZATION
             IN VEHICULAR CLOUD COMPUTING
             ---- 2x2 GRID  -  4 REGIONS  -  4 RSUs ----
=====================================================================

  Research papers synthesised:
   Paper 1 – DRL for Computation Offloading (DQN, latency-only)
   Paper 2 – Multi-Objective Optimization (NSGA-II, cost+latency)
   Paper 3 – DQN-Based Task Offloading in IoV (latency-only)
   Paper 4 – Vehicle Count Prediction (ML comparison)
   Paper 5 – Traffic Flow Prediction (LR + RF)
   Paper 6 – ARIMA Baseline for Traffic

  Our contribution:
   -> Use the WINNING ML models from the Voting Algorithm:
        - Linear Regression     -> traffic count prediction
        - Decision Tree         -> vehicle speed & type prediction
        - Linear Regression     -> signal strength prediction
        - Random Forest         -> task requirement prediction
   -> Multi-stage prediction pipeline:
        1. Predict next-timestamp vehicle count per region
        2. Predict vehicle specifications (speed, signal, type)
        3. Predict task requirements (cpu, data, deadline)
        4. Cost-optimise offloading with multi-objective function
   -> Compare against a baseline (heuristic, NO ML)

  Requirements:
      pip install pandas numpy scikit-learn matplotlib
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, accuracy_score, f1_score)
from collections import Counter

# =====================================================================
# CONFIG
# =====================================================================
TRAFFIC_CSV    = "vehicular_traffic_dataset.csv"
VEHICLE_CSV    = "vehicle_individual_dataset.csv"
TASK_CSV       = "task_offloading_dataset.csv"
OUTPUT_MAIN    = "simulation_results.png"
OUTPUT_TREND   = "cost_trend.png"

GRID_ROWS, GRID_COLS = 2, 2          # 2×2 grid
NUM_REGIONS          = GRID_ROWS * GRID_COLS   # 4 regions
REGION_IDS           = [1, 2, 3, 4]
RSU_MAP              = {1: 'RSU_1', 2: 'RSU_2', 3: 'RSU_3', 4: 'RSU_4'}
TRAIN_RATIO          = 0.80

np.random.seed(42)

print("=" * 72)
print("  ML-DRIVEN MULTI-OBJECTIVE COST & LATENCY OPTIMIZATION")
print("  2x2 Grid  -  4 Regions  -  4 RSUs  -  Vehicular Cloud Computing")
print("=" * 72)

# =====================================================================
# HELPER — timestamp → integer index
# =====================================================================
def ts_to_idx(ts_series):
    """Convert HH:MM strings to integer indices preserving chronological order."""
    unique_ts = sorted(ts_series.unique(),
                       key=lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    return ts_series.map({t: i for i, t in enumerate(unique_ts)}), unique_ts

# =====================================================================
# PHASE 1: DATA LOADING & 2×2 GRID SETUP
# =====================================================================
print("\n" + "-" * 72)
print("  PHASE 1 | Data Loading & 2x2 Grid Setup")
print("-" * 72)

# ── 1a. Traffic dataset ──
df_traffic = pd.read_csv(TRAFFIC_CSV)
df_traffic['ts_idx'], ts_labels = ts_to_idx(df_traffic['timestamp'])
df_traffic = df_traffic.sort_values(['ts_idx', 'region_id']).reset_index(drop=True)
print(f"  Traffic data    : {len(df_traffic):>6,} rows  |  {df_traffic['timestamp'].nunique()} timestamps x {NUM_REGIONS} regions")

# ── 1b. Vehicle individual dataset ──
df_vehicle = pd.read_csv(VEHICLE_CSV)
df_vehicle['ts_idx'], _ = ts_to_idx(df_vehicle['timestamp'])
le_vtype = LabelEncoder()
df_vehicle['vehicle_type_enc'] = le_vtype.fit_transform(df_vehicle['vehicle_type'])
df_vehicle = df_vehicle.sort_values(['ts_idx', 'region_id', 'vehicle_id']).reset_index(drop=True)
print(f"  Vehicle data    : {len(df_vehicle):>6,} rows  |  types = {list(le_vtype.classes_)}")

# ── 1c. Task offloading dataset ──
df_task = pd.read_csv(TASK_CSV)
df_task['ts_idx'], _ = ts_to_idx(df_task['timestamp'])
le_task_type = LabelEncoder()
df_task['task_type_enc'] = le_task_type.fit_transform(df_task['task_type'])
df_task['assigned_rsu'] = df_task['assigned_rsu'].fillna('None')
le_rsu = LabelEncoder()
df_task['assigned_rsu_enc'] = le_rsu.fit_transform(df_task['assigned_rsu'])
le_offload = LabelEncoder()
df_task['offload_target_enc'] = le_offload.fit_transform(df_task['offload_target'])
df_task = df_task.sort_values(['ts_idx', 'region_id', 'task_id']).reset_index(drop=True)
print(f"  Task data       : {len(df_task):>6,} rows  |  types = {list(le_task_type.classes_)}")

print(f"\n  Grid layout : {GRID_ROWS}x{GRID_COLS} = {NUM_REGIONS} regions")
for r in REGION_IDS:
    row, col = divmod(r - 1, GRID_COLS)
    print(f"    Region {r} -> grid({row},{col}) -> {RSU_MAP[r]}")


# =====================================================================
# PHASE 2: TRAIN WINNING ML MODELS (from Voting Algorithm)
#          ─── No voting re-run; each model is the direct winner ───
# =====================================================================
print("\n" + "-" * 72)
print("  PHASE 2 | Training Winning ML Models")
print("-" * 72)

metrics = {}   # store evaluation metrics


# ── 2a. TRAFFIC PREDICTION : Linear Regression ──────────────────────
# Predict: total_vehicles_present, moving_vehicles
# Features: ts_idx, region_id, lag-1/2/3 per target, vehicles_arrived, vehicles_left
print("\n  [Traffic] Linear Regression for vehicle count prediction ...")

for col in ['total_vehicles_present', 'moving_vehicles']:
    df_traffic[f'{col}_lag1'] = df_traffic.groupby('region_id')[col].shift(1)
    df_traffic[f'{col}_lag2'] = df_traffic.groupby('region_id')[col].shift(2)
    df_traffic[f'{col}_lag3'] = df_traffic.groupby('region_id')[col].shift(3)

df_traffic_clean = df_traffic.dropna().reset_index(drop=True)

TRAFFIC_FEATS = ['ts_idx', 'region_id', 'vehicles_arrived', 'vehicles_left',
                 'total_vehicles_present_lag1', 'total_vehicles_present_lag2',
                 'total_vehicles_present_lag3',
                 'moving_vehicles_lag1', 'moving_vehicles_lag2', 'moving_vehicles_lag3']
TRAFFIC_TGTS  = ['total_vehicles_present', 'moving_vehicles']

split_tr = int(len(df_traffic_clean) * TRAIN_RATIO)
X_tr_trf = df_traffic_clean[TRAFFIC_FEATS].values[:split_tr]
Y_tr_trf = df_traffic_clean[TRAFFIC_TGTS].values[:split_tr]
X_te_trf = df_traffic_clean[TRAFFIC_FEATS].values[split_tr:]
Y_te_trf = df_traffic_clean[TRAFFIC_TGTS].values[split_tr:]

scaler_trf = StandardScaler()
X_tr_trf_sc = scaler_trf.fit_transform(X_tr_trf)
X_te_trf_sc = scaler_trf.transform(X_te_trf)

model_traffic = LinearRegression()
model_traffic.fit(X_tr_trf_sc, Y_tr_trf)
Y_pred_trf = model_traffic.predict(X_te_trf_sc)

for i, tgt in enumerate(TRAFFIC_TGTS):
    r2 = r2_score(Y_te_trf[:, i], Y_pred_trf[:, i])
    mae = mean_absolute_error(Y_te_trf[:, i], Y_pred_trf[:, i])
    rmse = np.sqrt(mean_squared_error(Y_te_trf[:, i], Y_pred_trf[:, i]))
    metrics[f'traffic_{tgt}'] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"    {tgt:<28} R²={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}")

df_traffic_test = df_traffic_clean.iloc[split_tr:].reset_index(drop=True)


# ── 2b. VEHICLE SPEC PREDICTION ─────────────────────────────────────
# Decision Tree  → speed_kmh (R²=0.7028)
# Linear Reg     → signal_strength_dBm (R²=0.4360)
# Decision Tree  → vehicle_type (Accuracy=1.0, F1=1.0) — classification
print("\n  [Vehicle] Decision Tree (speed, type) + Linear Reg (signal) ...")

VEH_FEATS = ['ts_idx', 'region_id', 'vehicle_type_enc', 'is_static']

split_vh = int(len(df_vehicle) * TRAIN_RATIO)
X_tr_vh = df_vehicle[VEH_FEATS].values[:split_vh]
X_te_vh = df_vehicle[VEH_FEATS].values[split_vh:]

# Speed — Decision Tree Regressor
y_tr_speed = df_vehicle['speed_kmh'].values[:split_vh]
y_te_speed = df_vehicle['speed_kmh'].values[split_vh:]
model_speed = DecisionTreeRegressor(max_depth=12, random_state=42)
model_speed.fit(X_tr_vh, y_tr_speed)
y_pred_speed = model_speed.predict(X_te_vh)
r2_sp = r2_score(y_te_speed, y_pred_speed)
metrics['vehicle_speed'] = {'R2': r2_sp,
                            'MAE': mean_absolute_error(y_te_speed, y_pred_speed),
                            'RMSE': np.sqrt(mean_squared_error(y_te_speed, y_pred_speed))}
print(f"    speed_kmh            (DT)   R²={r2_sp:.4f}")

# Signal strength — Linear Regression
scaler_vh = StandardScaler()
X_tr_vh_sc = scaler_vh.fit_transform(X_tr_vh)
X_te_vh_sc = scaler_vh.transform(X_te_vh)

y_tr_sig = df_vehicle['signal_strength_dBm'].values[:split_vh]
y_te_sig = df_vehicle['signal_strength_dBm'].values[split_vh:]
model_signal = LinearRegression()
model_signal.fit(X_tr_vh_sc, y_tr_sig)
y_pred_sig = model_signal.predict(X_te_vh_sc)
r2_sg = r2_score(y_te_sig, y_pred_sig)
metrics['vehicle_signal'] = {'R2': r2_sg,
                             'MAE': mean_absolute_error(y_te_sig, y_pred_sig),
                             'RMSE': np.sqrt(mean_squared_error(y_te_sig, y_pred_sig))}
print(f"    signal_strength_dBm  (LR)   R²={r2_sg:.4f}")

# Vehicle type — Decision Tree Classifier
VEH_CLS_FEATS = ['ts_idx', 'region_id', 'is_static', 'speed_kmh', 'signal_strength_dBm']
X_tr_cls = df_vehicle[VEH_CLS_FEATS].values[:split_vh]
X_te_cls = df_vehicle[VEH_CLS_FEATS].values[split_vh:]
y_tr_vtype = df_vehicle['vehicle_type_enc'].values[:split_vh]
y_te_vtype = df_vehicle['vehicle_type_enc'].values[split_vh:]
model_vtype = DecisionTreeClassifier(max_depth=15, random_state=42)
model_vtype.fit(X_tr_cls, y_tr_vtype)
y_pred_vtype = model_vtype.predict(X_te_cls)
acc_vt = accuracy_score(y_te_vtype, y_pred_vtype)
f1_vt  = f1_score(y_te_vtype, y_pred_vtype, average='weighted')
metrics['vehicle_type'] = {'Accuracy': acc_vt, 'F1': f1_vt}
print(f"    vehicle_type         (DT)   Acc={acc_vt:.4f}  F1={f1_vt:.4f}")


# ── 2c. TASK REQUIREMENT PREDICTION : Random Forest ─────────────────
# Predict: cpu_cycles_M, data_size_KB, deadline_ms
print("\n  [Task] Random Forest for task requirement prediction ...")

TASK_TGTS = ['cpu_cycles_M', 'data_size_KB', 'deadline_ms']

# Lag features
for col in TASK_TGTS:
    df_task[f'{col}_lag1'] = df_task[col].shift(1)
    df_task[f'{col}_lag2'] = df_task[col].shift(2)
    df_task[f'{col}_lag3'] = df_task[col].shift(3)

# Rolling mean & std per task_type (window=10)
for col in TASK_TGTS:
    df_task[f'{col}_roll_mean'] = df_task.groupby('task_type_enc')[col].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df_task[f'{col}_roll_std'] = df_task.groupby('task_type_enc')[col].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).std().fillna(0))

# Previous timestamp aggregate mean
ts_agg = df_task.groupby('ts_idx')[TASK_TGTS].mean().shift(1).add_prefix('prev_ts_mean_')
df_task = df_task.merge(ts_agg, on='ts_idx', how='left').dropna().reset_index(drop=True)

TASK_FEATS = [
    'ts_idx', 'region_id', 'task_type_enc', 'offload_target_enc', 'assigned_rsu_enc',
    'exec_latency_ms', 'cost_usd',
    'cpu_cycles_M_lag1', 'cpu_cycles_M_lag2', 'cpu_cycles_M_lag3',
    'data_size_KB_lag1', 'data_size_KB_lag2', 'data_size_KB_lag3',
    'deadline_ms_lag1',  'deadline_ms_lag2',  'deadline_ms_lag3',
    'cpu_cycles_M_roll_mean', 'cpu_cycles_M_roll_std',
    'data_size_KB_roll_mean', 'data_size_KB_roll_std',
    'deadline_ms_roll_mean',  'deadline_ms_roll_std',
    'prev_ts_mean_cpu_cycles_M', 'prev_ts_mean_data_size_KB', 'prev_ts_mean_deadline_ms',
]

X_task = df_task[TASK_FEATS].values
Y_task = df_task[TASK_TGTS].values

split_tk = int(len(X_task) * TRAIN_RATIO)
X_tr_tk, X_te_tk = X_task[:split_tk], X_task[split_tk:]
Y_tr_tk, Y_te_tk = Y_task[:split_tk], Y_task[split_tk:]

model_task = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_task.fit(X_tr_tk, Y_tr_tk)
Y_pred_tk = model_task.predict(X_te_tk)

for i, tgt in enumerate(TASK_TGTS):
    r2 = r2_score(Y_te_tk[:, i], Y_pred_tk[:, i])
    mae = mean_absolute_error(Y_te_tk[:, i], Y_pred_tk[:, i])
    rmse = np.sqrt(mean_squared_error(Y_te_tk[:, i], Y_pred_tk[:, i]))
    metrics[f'task_{tgt}'] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
    print(f"    {tgt:<20} R²={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}")

df_task_test = df_task.iloc[split_tk:].reset_index(drop=True)


# =====================================================================
# PHASE 3: 2×2 GRID TIME-STEP SIMULATION
#          ─── Predict traffic → vehicles → tasks → optimise ───
# =====================================================================
print("\n" + "-" * 72)
print("  PHASE 3 | 2x2 Grid Time-Step Simulation")
print("-" * 72)

# -- 3a. Infrastructure parameters (calibrated from dataset) ----------
RSU_COST_PER_CPU   = 0.00002     # USD per M cpu cycles
CLOUD_COST_PER_CPU = 0.00005     # USD per M cpu cycles
BW_COST_PER_KB     = 0.0000001   # bandwidth cost component

INFRA = {
    'Local': {
        'cpu_speed_M_per_ms':  3.0,
        'bw_KB_per_ms':        20.0,
        'cost_per_cpu_M':      0.00001,
        'bw_cost_per_KB':      0.00000005,
        'base_latency_ms':     4,
    },
    'RSU': {
        'cpu_speed_M_per_ms':  18.0,
        'bw_KB_per_ms':        80.0,
        'cost_per_cpu_M':      RSU_COST_PER_CPU,
        'bw_cost_per_KB':      BW_COST_PER_KB,
        'base_latency_ms':     8,
    },
    'Cloud': {
        'cpu_speed_M_per_ms':  50.0,
        'bw_KB_per_ms':        8.0,
        'cost_per_cpu_M':      CLOUD_COST_PER_CPU,
        'bw_cost_per_KB':      BW_COST_PER_KB * 3,
        'base_latency_ms':     80,
    },
}

# RSU capacity: max concurrent tasks per RSU (scales with vehicle density)
RSU_BASE_CAPACITY = 50    # base concurrent tasks

# -- 3b. Dynamic task-type weights (from research papers) -------------
# w1 = latency priority, w2 = cost priority, w1 + w2 = 1
TASK_WEIGHTS = {
    'autonomous_alert': {'w1': 0.90, 'w2': 0.10},
    'navigation':       {'w1': 0.65, 'w2': 0.35},
    'video_stream':     {'w1': 0.25, 'w2': 0.75},
    'infotainment':     {'w1': 0.15, 'w2': 0.85},
    'sensor_upload':    {'w1': 0.35, 'w2': 0.65},
}

print(f"\n  Infrastructure pricing:")
print(f"    RSU   : ${RSU_COST_PER_CPU:.6f}/M-cpu  |  base latency  8 ms")
print(f"    Cloud : ${CLOUD_COST_PER_CPU:.6f}/M-cpu  |  base latency 80 ms")
print(f"    Cloud/RSU cost ratio = {CLOUD_COST_PER_CPU / RSU_COST_PER_CPU:.1f}x")
print(f"    RSU base capacity    = {RSU_BASE_CAPACITY} concurrent tasks")


# -- 3c. Helper functions -------------------------------------------------
def signal_bandwidth_factor(signal_dBm):
    """Scale bandwidth based on signal strength (Paper 1/3: transmission rate varies)."""
    if signal_dBm > -60:
        return 1.0        # strong signal → full bandwidth
    elif signal_dBm > -75:
        return 0.70       # medium -> 70%
    else:
        return 0.40       # weak -> 40%

def speed_handoff_penalty(speed_kmh, target_name='RSU'):
    """Add latency penalty for fast-moving vehicles.
    RSU: risk of leaving RSU coverage zone.
    Local (V2V): risk of losing connection to the nearby vehicle — both are
    moving, so relative speed matters even more."""
    if target_name == 'Local':        # V2V: higher penalty (both vehicles move)
        if speed_kmh > 70:
            return 25.0
        elif speed_kmh > 50:
            return 12.0
        return 0.0
    else:                             # RSU: fixed infrastructure
        if speed_kmh > 70:
            return 20.0
        elif speed_kmh > 50:
            return 10.0
        return 0.0

def compute_latency(cpu_M, data_KB, infra, signal_dBm=-65.0, speed_kmh=40.0,
                    target_name='RSU'):
    """
    Compute total latency:
        T = T_base + T_upload + T_compute + T_handoff
    (Papers 1, 2, 3 unified)

    All three targets (Local/V2V, RSU, Cloud) involve wireless transmission,
    so signal strength always affects effective bandwidth.
    Local = V2V (offload to a nearby vehicle's spare CPU via DSRC/C-V2X).
    """
    # Signal affects all wireless links: V2V (Local), RSU, and Cloud uplink
    bw_factor = signal_bandwidth_factor(signal_dBm)
    effective_bw = infra['bw_KB_per_ms'] * bw_factor

    t_upload  = data_KB / effective_bw
    t_compute = cpu_M / infra['cpu_speed_M_per_ms']

    # Handoff penalty for RSU (leaving coverage) and Local/V2V (both vehicles moving)
    if target_name in ('RSU', 'Local'):
        t_handoff = speed_handoff_penalty(speed_kmh, target_name)
    else:
        t_handoff = 0.0

    return infra['base_latency_ms'] + t_upload + t_compute + t_handoff

def compute_cost(cpu_M, data_KB, infra):
    """
    Cost = α × CPU_used + β × BW_used   (Paper 2 equation)
    """
    return infra['cost_per_cpu_M'] * cpu_M + infra['bw_cost_per_KB'] * data_KB

def optimise_offloading(cpu_M, data_KB, deadline_ms, task_type,
                        signal_dBm, speed_kmh, rsu_load_ratio,
                        max_lat, max_cst):
    """
    Multi-objective optimisation:
        F = w1 × Norm_Latency + w2 × Norm_Cost
    with deadline penalty & RSU capacity awareness.
    """
    w = TASK_WEIGHTS.get(task_type, {'w1': 0.5, 'w2': 0.5})
    w1, w2 = w['w1'], w['w2']

    best_F, best_target, best_lat, best_cst = float('inf'), 'RSU', 0, 0

    for infra_name, infra in INFRA.items():
        lat = compute_latency(cpu_M, data_KB, infra, signal_dBm, speed_kmh, infra_name)
        cst = compute_cost(cpu_M, data_KB, infra)

        # Normalise to [0, 1]
        norm_lat = lat / max_lat if max_lat > 0 else 0
        norm_cst = cst / max_cst if max_cst > 0 else 0

        # Multi-objective score
        F = w1 * norm_lat + w2 * norm_cst

        # Penalty if deadline missed (Paper 1 reward = -T)
        if lat > deadline_ms:
            F += 10.0

        # RSU overload penalty: if RSU is congested, penalise RSU choice
        if infra_name == 'RSU' and rsu_load_ratio > 0.85:
            F += 2.0 * (rsu_load_ratio - 0.85)

        if F < best_F:
            best_F      = F
            best_target = infra_name
            best_lat    = lat
            best_cst    = cst

    deadline_met = 1 if best_lat <= deadline_ms else 0
    return best_target, best_lat, best_cst, best_F, deadline_met


# -- 3d. Run the simulation -----------------------------------------------
print(f"\n  Simulating on {len(df_task_test):,} test tasks across "
      f"{df_task_test['ts_idx'].nunique()} timestamps ...")

# Get predicted task requirements
pred_cpu      = np.clip(Y_pred_tk[:, 0], 50, 1500)
pred_data     = np.clip(Y_pred_tk[:, 1], 10, 2000)
pred_deadline = np.clip(Y_pred_tk[:, 2], 50, 800)

actual_cpu      = Y_te_tk[:, 0]
actual_data     = Y_te_tk[:, 1]
actual_deadline = Y_te_tk[:, 2]

task_types   = df_task_test['task_type'].values
region_ids   = df_task_test['region_id'].values
ts_indices   = df_task_test['ts_idx'].values
vehicle_ids  = df_task_test['vehicle_id'].values

n_tasks = len(pred_cpu)

# Build lookup: vehicle_id → (speed, signal) from vehicle test set
# Use the ML-predicted values for test-set vehicles
veh_speed_map  = dict(zip(df_vehicle['vehicle_id'].values[split_vh:],
                          y_pred_speed))
veh_signal_map = dict(zip(df_vehicle['vehicle_id'].values[split_vh:],
                          y_pred_sig))
# Fallback from training set for vehicles not in test
for vid, spd, sig in zip(df_vehicle['vehicle_id'].values[:split_vh],
                         df_vehicle['speed_kmh'].values[:split_vh],
                         df_vehicle['signal_strength_dBm'].values[:split_vh]):
    if vid not in veh_speed_map:
        veh_speed_map[vid] = spd
        veh_signal_map[vid] = sig

# Build per-region per-timestamp traffic prediction lookup
# from the traffic test set
traffic_pred_lookup = {}  # (ts_idx, region_id) → predicted total_vehicles
for idx in range(len(df_traffic_test)):
    key = (df_traffic_test.iloc[idx]['ts_idx'], df_traffic_test.iloc[idx]['region_id'])
    traffic_pred_lookup[key] = max(Y_pred_trf[idx, 0], 1)  # at least 1 vehicle

# Pre-compute normalisation constants
all_lat, all_cst = [], []
for infra_name, infra in INFRA.items():
    for i in range(min(n_tasks, 5000)):
        sig = veh_signal_map.get(vehicle_ids[i], -65.0)
        spd = veh_speed_map.get(vehicle_ids[i], 40.0)
        all_lat.append(compute_latency(pred_cpu[i], pred_data[i], infra, sig, spd, infra_name))
        all_cst.append(compute_cost(pred_cpu[i], pred_data[i], infra))
max_lat = max(all_lat) if all_lat else 1
max_cst = max(all_cst) if all_cst else 1

# ---- OPTIMISED (ML-driven) decisions ----
opt_latency      = np.zeros(n_tasks)
opt_cost         = np.zeros(n_tasks)
opt_target       = np.empty(n_tasks, dtype=object)
opt_F            = np.zeros(n_tasks)
opt_deadline_met = np.zeros(n_tasks, dtype=int)

# Track RSU load per (timestamp, region)
rsu_task_counts = Counter()   # (ts_idx, region_id) → count of tasks sent to RSU

for i in range(n_tasks):
    ts   = ts_indices[i]
    rid  = region_ids[i]
    vid  = vehicle_ids[i]

    sig  = veh_signal_map.get(vid, -65.0)
    spd  = veh_speed_map.get(vid, 40.0)
    ttype = task_types[i]

    # RSU load ratio for this region at this timestamp
    predicted_vehicles = traffic_pred_lookup.get((ts, rid), 50)
    rsu_cap = RSU_BASE_CAPACITY + predicted_vehicles * 0.3   # dynamic capacity
    current_rsu_tasks = rsu_task_counts[(ts, rid)]
    rsu_load_ratio = current_rsu_tasks / rsu_cap if rsu_cap > 0 else 1.0

    target, lat, cst, f_val, dl_met = optimise_offloading(
        pred_cpu[i], pred_data[i], pred_deadline[i],
        ttype, sig, spd, rsu_load_ratio, max_lat, max_cst)

    opt_latency[i]      = lat
    opt_cost[i]         = cst
    opt_target[i]       = target
    opt_F[i]            = f_val
    opt_deadline_met[i] = dl_met

    if target == 'RSU':
        rsu_task_counts[(ts, rid)] += 1

# ---- BASELINE (No-ML heuristic) decisions ----
# Static rule: always try RSU first, overflow to Cloud. No signal/speed awareness.
# No task-type-aware weights. Uses ACTUAL values (no prediction).
print("  Running baseline (no-ML heuristic) ...")

base_latency      = np.zeros(n_tasks)
base_cost         = np.zeros(n_tasks)
base_target       = np.empty(n_tasks, dtype=object)
base_deadline_met = np.zeros(n_tasks, dtype=int)

BASELINE_RSU_CAP  = RSU_BASE_CAPACITY   # fixed capacity, no dynamic scaling
base_rsu_counts   = Counter()

for i in range(n_tasks):
    ts  = ts_indices[i]
    rid = region_ids[i]

    cpu_val  = actual_cpu[i]
    data_val = actual_data[i]
    dl_val   = actual_deadline[i]

    # Baseline uses default signal/speed (no prediction)
    default_sig = -65.0
    default_spd = 40.0

    # Simple rule: try RSU; if overloaded, go Cloud
    if base_rsu_counts[(ts, rid)] < BASELINE_RSU_CAP:
        chosen = 'RSU'
    else:
        chosen = 'Cloud'

    infra = INFRA[chosen]
    lat = compute_latency(cpu_val, data_val, infra, default_sig, default_spd, chosen)
    cst = compute_cost(cpu_val, data_val, infra)

    base_latency[i]      = lat
    base_cost[i]         = cst
    base_target[i]       = chosen
    base_deadline_met[i] = 1 if lat <= dl_val else 0

    if chosen == 'RSU':
        base_rsu_counts[(ts, rid)] += 1


# =====================================================================
# PHASE 4: RESULTS COMPARISON & ANALYSIS
# =====================================================================
print("\n" + "-" * 72)
print("  PHASE 4 | Results -- Baseline (No-ML) vs ML-Optimised")
print("-" * 72)

baseline_total_cost   = base_cost.sum()
optimised_total_cost  = opt_cost.sum()
cost_savings_pct      = (1 - optimised_total_cost / baseline_total_cost) * 100 if baseline_total_cost > 0 else 0

baseline_avg_lat  = base_latency.mean()
optimised_avg_lat = opt_latency.mean()
latency_change_pct = (1 - optimised_avg_lat / baseline_avg_lat) * 100 if baseline_avg_lat > 0 else 0

baseline_dl_rate  = base_deadline_met.mean() * 100
optimised_dl_rate = opt_deadline_met.mean() * 100

print(f"""
  +-------------------------------+-----------------+-----------------+--------------+
  | Metric                        | Baseline (No-ML)| ML-Optimised    | Improvement  |
  +-------------------------------+-----------------+-----------------+--------------+
  | Total Cost (USD)              | ${baseline_total_cost:>12.4f}  | ${optimised_total_cost:>12.4f}  | {cost_savings_pct:>+9.2f}%  |
  | Avg Latency (ms)              | {baseline_avg_lat:>14.2f}  | {optimised_avg_lat:>14.2f}  | {latency_change_pct:>+9.2f}%  |
  | Avg Cost/Task (USD)           | ${base_cost.mean():>12.6f}  | ${opt_cost.mean():>12.6f}  | {cost_savings_pct:>+9.2f}%  |
  | Deadline Met Rate (%)         | {baseline_dl_rate:>13.2f}% | {optimised_dl_rate:>13.2f}% | {optimised_dl_rate - baseline_dl_rate:>+9.2f}pp |
  +-------------------------------+-----------------+-----------------+--------------+
""")

# Per-task-type breakdown
unique_types = sorted(df_task_test['task_type'].unique())
print("  Per-Task-Type Breakdown:")
print(f"  {'Task Type':<22} {'Base Cost':>12} {'Opt Cost':>12} {'Savings':>9} "
      f"{'Base Lat':>9} {'Opt Lat':>9} {'Opt Target':>10}")
print("  " + "-" * 90)

task_type_stats = {}
for tt in unique_types:
    mask = task_types == tt
    b_cost   = base_cost[mask].sum()
    o_cost   = opt_cost[mask].sum()
    savings  = (1 - o_cost / b_cost) * 100 if b_cost > 0 else 0
    b_lat    = base_latency[mask].mean()
    o_lat    = opt_latency[mask].mean()
    b_dl     = base_deadline_met[mask].mean() * 100
    o_dl     = opt_deadline_met[mask].mean() * 100
    top_tgt  = Counter(opt_target[mask]).most_common(1)[0]

    task_type_stats[tt] = {
        'baseline_cost': b_cost, 'opt_cost': o_cost,
        'savings_pct': savings, 'primary_target': top_tgt[0],
        'count': mask.sum(),
        'baseline_latency': b_lat, 'opt_latency': o_lat,
        'baseline_dl_rate': b_dl,  'opt_dl_rate': o_dl,
    }
    print(f"  {tt:<22} ${b_cost:>10.4f} ${o_cost:>10.4f} {savings:>+7.1f}%  "
          f"{b_lat:>7.1f}ms {o_lat:>7.1f}ms  -> {top_tgt[0]}")

# Per-region breakdown
print(f"\n  Per-Region Breakdown:")
print(f"  {'Region':>8} {'RSU':>6} {'Base Cost':>12} {'Opt Cost':>12} {'Savings':>9} "
      f"{'Base Lat':>9} {'Opt Lat':>9}")
print("  " + "-" * 72)
region_stats = {}
for rid in REGION_IDS:
    mask = region_ids == rid
    b_c = base_cost[mask].sum()
    o_c = opt_cost[mask].sum()
    sav = (1 - o_c / b_c) * 100 if b_c > 0 else 0
    b_l = base_latency[mask].mean()
    o_l = opt_latency[mask].mean()
    region_stats[rid] = {'baseline_cost': b_c, 'opt_cost': o_c,
                         'savings_pct': sav, 'baseline_lat': b_l, 'opt_lat': o_l,
                         'count': mask.sum()}
    print(f"  {rid:>8} {RSU_MAP[rid]:>6} ${b_c:>10.4f} ${o_c:>10.4f} {sav:>+7.1f}%  "
          f"{b_l:>7.1f}ms {o_l:>7.1f}ms")

# Offloading distribution
print(f"\n  Offloading Distribution:")
baseline_dist = Counter(base_target)
opt_dist      = Counter(opt_target)
print(f"    {'Target':<10} {'Baseline':>10} {'Optimised':>10} {'Change':>10}")
for t in ['Local', 'RSU', 'Cloud']:
    b = baseline_dist.get(t, 0)
    o = opt_dist.get(t, 0)
    pct = ((o - b) / b * 100) if b > 0 else (100 if o > 0 else 0)
    print(f"    {t:<10} {b:>10,} {o:>10,} {pct:>+9.1f}%")


# =====================================================================
# PHASE 5: COMPREHENSIVE VISUALISATION  (6 rows × 3 cols)
# =====================================================================
print("\n" + "-" * 72)
print("  PHASE 5 | Generating Visualisations")
print("-" * 72)

# ─── Publication-Ready Color Palette ─────────────────────────────────
# Pastel tones with selective vibrant accents for readability
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

# Core comparison colors (pastel yet distinct)
CLR_BASELINE  = '#E07A5F'   # warm salmon / terracotta
CLR_OPTIMISED = '#81B29A'   # sage green
CLR_ACCENT1   = '#3D405B'   # charcoal indigo
CLR_ACCENT2   = '#F2CC8F'   # muted gold / sand
CLR_ACCENT3   = '#7FB3D3'   # soft steel blue
CLR_ACCENT4   = '#C9A9D2'   # lavender
CLR_SCATTER1  = '#4A7C94'   # teal
CLR_SCATTER2  = '#D4726A'   # dusty rose
CLR_SCATTER3  = '#6A9F58'   # olive green
CLR_GRIDLINE  = '#D0D0D0'   # light gray
CLR_SPINE     = '#AAAAAA'   # medium gray
CLR_TEXT      = '#2B2B2B'   # near-black

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.titlesize': 16,
})

fig = plt.figure(figsize=(24, 34))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(7, 3, figure=fig, hspace=0.58, wspace=0.36)


def style_ax(ax, title):
    """Apply clean, publication-ready styling to an axis."""
    ax.set_facecolor('#FAFAFA')
    ax.set_title(title, color=CLR_TEXT, fontsize=11, fontweight='bold', pad=12)
    ax.tick_params(colors=CLR_TEXT, labelsize=8.5, direction='out', length=3)
    ax.grid(True, linestyle='--', linewidth=0.4, color=CLR_GRIDLINE, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor(CLR_SPINE)
        spine.set_linewidth(0.6)


def style_legend(ax, **kwargs):
    """Consistent legend styling for publication."""
    defaults = dict(facecolor='white', edgecolor=CLR_SPINE, labelcolor=CLR_TEXT,
                    fontsize=8.5, framealpha=0.9)
    defaults.update(kwargs)
    return ax.legend(**defaults)


# ── Row 0: 2x2 Grid Heatmap (Vehicle Density) ───────────────────────
ax = fig.add_subplot(gs[0, 0])
style_ax(ax, 'Predicted Vehicle Density\n(2x2 Grid, per Region)')
ax.grid(False)

grid_data = np.zeros((GRID_ROWS, GRID_COLS))
for rid in REGION_IDS:
    row, col = divmod(rid - 1, GRID_COLS)
    mask = df_traffic_test['region_id'] == rid
    if mask.sum() > 0:
        idx_mask = np.where(mask)[0]
        grid_data[row, col] = Y_pred_trf[idx_mask, 0].mean()

im = ax.imshow(grid_data, cmap='YlOrBr', aspect='equal', vmin=0)
for i in range(GRID_ROWS):
    for j in range(GRID_COLS):
        rid = i * GRID_COLS + j + 1
        val = grid_data[i, j]
        text_color = 'white' if val > grid_data.mean() else CLR_TEXT
        ax.text(j, i, f'R{rid}\n{val:.0f} vehicles\n{RSU_MAP[rid]}',
                ha='center', va='center', fontsize=9, color=text_color,
                fontweight='bold')
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label('Avg. Vehicle Count', fontsize=9, color=CLR_TEXT)
cbar.ax.tick_params(colors=CLR_TEXT, labelsize=8)


# ── Row 0: Traffic Prediction Accuracy ───────────────────────────────
ax = fig.add_subplot(gs[0, 1])
style_ax(ax, 'Traffic Prediction Accuracy\n(Linear Regression)')

r2_list = [metrics['traffic_total_vehicles_present']['R2'],
           metrics['traffic_moving_vehicles']['R2']]
labels  = ['Total Vehicles\nPresent', 'Moving\nVehicles']
bars = ax.bar(labels, r2_list,
              color=[CLR_ACCENT3, CLR_OPTIMISED], alpha=0.92,
              edgecolor='white', linewidth=0.8, width=0.55)
for bar, v in zip(bars, r2_list):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'R2 = {v:.4f}', ha='center', va='bottom', fontsize=9,
            color=CLR_TEXT, fontweight='bold')
ax.set_ylim(0, 1.12)
ax.set_ylabel('R2 Score', color=CLR_TEXT, fontsize=9)
ax.axhline(1.0, color=CLR_SPINE, lw=0.8, ls='--', alpha=0.7)


# ── Row 0: Vehicle & Task Model Scores ───────────────────────────────
ax = fig.add_subplot(gs[0, 2])
style_ax(ax, 'ML Model Performance\n(Winning Models from Voting)')

model_labels = ['Speed\n(DT)', 'Signal\n(LR)', 'Type\n(DT)',
                'CPU Cycles\n(RF)', 'Data Size\n(RF)', 'Deadline\n(RF)']
model_scores = [
    metrics['vehicle_speed']['R2'],
    metrics['vehicle_signal']['R2'],
    metrics['vehicle_type']['Accuracy'],
    metrics['task_cpu_cycles_M']['R2'],
    metrics['task_data_size_KB']['R2'],
    metrics['task_deadline_ms']['R2'],
]
colors_m = [CLR_ACCENT2, CLR_ACCENT2, CLR_ACCENT2,
            CLR_ACCENT4, CLR_ACCENT4, CLR_ACCENT4]
bars = ax.bar(model_labels, model_scores, color=colors_m, alpha=0.92,
              edgecolor='white', linewidth=0.8, width=0.65)
for bar, v in zip(bars, model_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'{v:.3f}', ha='center', va='bottom', fontsize=7.5,
            color=CLR_TEXT, fontweight='bold')
ax.set_ylim(0, 1.18)
ax.set_ylabel('Score (R2 / Accuracy)', color=CLR_TEXT, fontsize=9)
ax.axhline(1.0, color=CLR_SPINE, lw=0.8, ls='--', alpha=0.7)

legend_patches = [Patch(facecolor=CLR_ACCENT2, edgecolor='white',
                        label='Vehicle Models'),
                  Patch(facecolor=CLR_ACCENT4, edgecolor='white',
                        label='Task Models (Random Forest)')]
style_legend(ax, handles=legend_patches, loc='upper right')


# ── Row 1: Cost Comparison ───────────────────────────────────────────
x_pos = np.arange(len(unique_types))
bar_w = 0.35

# 1a: Total cost per task type
ax = fig.add_subplot(gs[1, 0])
style_ax(ax, 'Total Cost by Task Type\n(Baseline vs ML-Optimised)')
ax.bar(x_pos - bar_w/2,
       [task_type_stats[t]['baseline_cost'] for t in unique_types],
       bar_w, label='Baseline (No-ML)', color=CLR_BASELINE, alpha=0.90,
       edgecolor='white', lw=0.8)
ax.bar(x_pos + bar_w/2,
       [task_type_stats[t]['opt_cost'] for t in unique_types],
       bar_w, label='ML-Optimised', color=CLR_OPTIMISED, alpha=0.90,
       edgecolor='white', lw=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([t.replace('_', '\n') for t in unique_types],
                   fontsize=7.5, color=CLR_TEXT)
ax.set_ylabel('Total Cost (USD)', color=CLR_TEXT, fontsize=9)
style_legend(ax)

# 1b: Average latency per task type
ax = fig.add_subplot(gs[1, 1])
style_ax(ax, 'Avg Latency by Task Type\n(Baseline vs ML-Optimised)')
ax.bar(x_pos - bar_w/2,
       [task_type_stats[t]['baseline_latency'] for t in unique_types],
       bar_w, label='Baseline (No-ML)', color=CLR_BASELINE, alpha=0.90,
       edgecolor='white', lw=0.8)
ax.bar(x_pos + bar_w/2,
       [task_type_stats[t]['opt_latency'] for t in unique_types],
       bar_w, label='ML-Optimised', color=CLR_OPTIMISED, alpha=0.90,
       edgecolor='white', lw=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([t.replace('_', '\n') for t in unique_types],
                   fontsize=7.5, color=CLR_TEXT)
ax.set_ylabel('Avg Latency (ms)', color=CLR_TEXT, fontsize=9)
style_legend(ax)

# 1c: Cost savings % per task type
ax = fig.add_subplot(gs[1, 2])
style_ax(ax, 'Cost Savings by Task Type (%)')
savings_vals = [task_type_stats[t]['savings_pct'] for t in unique_types]
colors_bar = [CLR_OPTIMISED if v > 0 else CLR_BASELINE for v in savings_vals]
bars = ax.barh([t.replace('_', '\n') for t in unique_types], savings_vals,
               color=colors_bar, alpha=0.88, edgecolor='white', lw=0.8,
               height=0.55)
for bar, v in zip(bars, savings_vals):
    xpos = bar.get_width() + 1.0 if v >= 0 else bar.get_width() - 4
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f'{v:+.1f}%', va='center', fontsize=9, color=CLR_TEXT,
            fontweight='bold')
ax.set_xlabel('Cost Savings (%)', color=CLR_TEXT, fontsize=9)
ax.axvline(0, color=CLR_TEXT, lw=0.8, alpha=0.4)


# ── Row 2: Offloading Distribution + Dynamic Weights ────────────────
pie_colors_pub = {'RSU': CLR_ACCENT3, 'Cloud': CLR_BASELINE, 'Local': CLR_OPTIMISED}

# 2a: Offloading pie (Baseline)
ax = fig.add_subplot(gs[2, 0])
style_ax(ax, 'Offloading Distribution\n(Baseline - No ML)')
ax.grid(False)
b_labels = [k for k in baseline_dist if baseline_dist[k] > 0]
b_sizes  = [baseline_dist[k] for k in b_labels]
wedges, texts, autotexts = ax.pie(
    b_sizes, labels=b_labels, autopct='%1.1f%%',
    colors=[pie_colors_pub.get(l, '#CCC') for l in b_labels],
    textprops={'color': CLR_TEXT, 'fontsize': 10, 'fontweight': 'bold'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    startangle=90)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight('bold')

# 2b: Offloading pie (Optimised)
ax = fig.add_subplot(gs[2, 1])
style_ax(ax, 'Offloading Distribution\n(ML-Optimised)')
ax.grid(False)
o_labels = [k for k in opt_dist if opt_dist[k] > 0]
o_sizes  = [opt_dist[k] for k in o_labels]
wedges, texts, autotexts = ax.pie(
    o_sizes, labels=o_labels, autopct='%1.1f%%',
    colors=[pie_colors_pub.get(l, '#CCC') for l in o_labels],
    textprops={'color': CLR_TEXT, 'fontsize': 10, 'fontweight': 'bold'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    startangle=90)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight('bold')

# 2c: Dynamic weight visualisation
ax = fig.add_subplot(gs[2, 2])
style_ax(ax, 'Task-Type Optimization Weights\n(w1 = Latency, w2 = Cost)')
tt_list = list(TASK_WEIGHTS.keys())
w1_vals = [TASK_WEIGHTS[t]['w1'] for t in tt_list]
w2_vals = [TASK_WEIGHTS[t]['w2'] for t in tt_list]
y_pos_w = np.arange(len(tt_list))
ax.barh(y_pos_w, w1_vals, 0.4, label='w1 (Latency Priority)',
        color='#E07A5F', alpha=0.88, edgecolor='white', lw=0.8)
ax.barh(y_pos_w, [-w for w in w2_vals], 0.4, label='w2 (Cost Priority)',
        color='#7FB3D3', alpha=0.88, edgecolor='white', lw=0.8)
ax.set_yticks(y_pos_w)
ax.set_yticklabels([t.replace('_', '\n') for t in tt_list],
                   fontsize=8.5, color=CLR_TEXT)
ax.set_xlabel('Weight Value', color=CLR_TEXT, fontsize=9)
ax.axvline(0, color=CLR_TEXT, lw=0.8, alpha=0.4)
style_legend(ax, loc='lower right')
ax.set_xlim(-1.05, 1.05)
for i, (w1v, w2v) in enumerate(zip(w1_vals, w2_vals)):
    ax.text(w1v + 0.03, i, f'{w1v:.2f}', va='center', fontsize=8,
            color='#B84A3A', fontweight='bold')
    ax.text(-w2v - 0.03, i, f'{w2v:.2f}', va='center', ha='right',
            fontsize=8, color='#3A7DA3', fontweight='bold')


# ── Row 3: Per-Region Analysis ───────────────────────────────────────
r_pos = np.arange(len(REGION_IDS))

# 3a: Cost by region
ax = fig.add_subplot(gs[3, 0])
style_ax(ax, 'Total Cost by Region\n(Baseline vs ML-Optimised)')
ax.bar(r_pos - bar_w/2,
       [region_stats[r]['baseline_cost'] for r in REGION_IDS],
       bar_w, label='Baseline (No-ML)', color=CLR_BASELINE, alpha=0.90,
       edgecolor='white', lw=0.8)
ax.bar(r_pos + bar_w/2,
       [region_stats[r]['opt_cost'] for r in REGION_IDS],
       bar_w, label='ML-Optimised', color=CLR_OPTIMISED, alpha=0.90,
       edgecolor='white', lw=0.8)
ax.set_xticks(r_pos)
ax.set_xticklabels([f'Region {r}\n({RSU_MAP[r]})' for r in REGION_IDS],
                   fontsize=8, color=CLR_TEXT)
ax.set_ylabel('Total Cost (USD)', color=CLR_TEXT, fontsize=9)
style_legend(ax)

# 3b: Latency by region
ax = fig.add_subplot(gs[3, 1])
style_ax(ax, 'Avg Latency by Region\n(Baseline vs ML-Optimised)')
ax.bar(r_pos - bar_w/2,
       [region_stats[r]['baseline_lat'] for r in REGION_IDS],
       bar_w, label='Baseline (No-ML)', color=CLR_BASELINE, alpha=0.90,
       edgecolor='white', lw=0.8)
ax.bar(r_pos + bar_w/2,
       [region_stats[r]['opt_lat'] for r in REGION_IDS],
       bar_w, label='ML-Optimised', color=CLR_OPTIMISED, alpha=0.90,
       edgecolor='white', lw=0.8)
ax.set_xticks(r_pos)
ax.set_xticklabels([f'Region {r}\n({RSU_MAP[r]})' for r in REGION_IDS],
                   fontsize=8, color=CLR_TEXT)
ax.set_ylabel('Avg Latency (ms)', color=CLR_TEXT, fontsize=9)
style_legend(ax)

# 3c: Deadline met rate by region
ax = fig.add_subplot(gs[3, 2])
style_ax(ax, 'Deadline Met Rate by Region (%)\n(Baseline vs ML-Optimised)')
b_dl_regions = []
o_dl_regions = []
for rid in REGION_IDS:
    mask = region_ids == rid
    b_dl_regions.append(base_deadline_met[mask].mean() * 100)
    o_dl_regions.append(opt_deadline_met[mask].mean() * 100)
ax.bar(r_pos - bar_w/2, b_dl_regions, bar_w, label='Baseline (No-ML)',
       color=CLR_BASELINE, alpha=0.90, edgecolor='white', lw=0.8)
ax.bar(r_pos + bar_w/2, o_dl_regions, bar_w, label='ML-Optimised',
       color=CLR_OPTIMISED, alpha=0.90, edgecolor='white', lw=0.8)
ax.set_xticks(r_pos)
ax.set_xticklabels([f'Region {r}\n({RSU_MAP[r]})' for r in REGION_IDS],
                   fontsize=8, color=CLR_TEXT)
ax.set_ylabel('Deadline Met (%)', color=CLR_TEXT, fontsize=9)
ax.set_ylim(0, 110)
style_legend(ax)


# ── Row 4: Scatter Comparisons ───────────────────────────────────────
sample_idx = np.random.RandomState(42).choice(n_tasks, min(2000, n_tasks), replace=False)

# 4a: Cost scatter
ax = fig.add_subplot(gs[4, 0])
style_ax(ax, 'Per-Task Cost Comparison\n(Baseline vs ML-Optimised)')
ax.scatter(base_cost[sample_idx], opt_cost[sample_idx],
           alpha=0.40, s=10, color=CLR_SCATTER1, edgecolors='none')
mn = min(base_cost.min(), opt_cost.min())
mx = max(base_cost.max(), opt_cost.max())
ax.plot([mn, mx], [mn, mx], '--', color=CLR_TEXT, lw=1, alpha=0.5,
        label='No-change line')
ax.set_xlabel('Baseline Cost (USD)', color=CLR_TEXT, fontsize=9)
ax.set_ylabel('ML-Optimised Cost (USD)', color=CLR_TEXT, fontsize=9)
style_legend(ax)

# 4b: Latency scatter
ax = fig.add_subplot(gs[4, 1])
style_ax(ax, 'Per-Task Latency Comparison\n(Baseline vs ML-Optimised)')
ax.scatter(base_latency[sample_idx], opt_latency[sample_idx],
           alpha=0.40, s=10, color=CLR_SCATTER2, edgecolors='none')
mn = min(base_latency.min(), opt_latency.min())
mx = max(base_latency.max(), opt_latency.max())
ax.plot([mn, mx], [mn, mx], '--', color=CLR_TEXT, lw=1, alpha=0.5,
        label='No-change line')
ax.set_xlabel('Baseline Latency (ms)', color=CLR_TEXT, fontsize=9)
ax.set_ylabel('ML-Optimised Latency (ms)', color=CLR_TEXT, fontsize=9)
style_legend(ax)

# 4c: Actual vs Predicted cpu_cycles (RF quality)
ax = fig.add_subplot(gs[4, 2])
style_ax(ax, 'Random Forest Prediction Quality\nActual vs Predicted cpu_cycles_M')
ax.scatter(Y_te_tk[sample_idx, 0], Y_pred_tk[sample_idx, 0],
           alpha=0.40, s=10, color=CLR_SCATTER3, edgecolors='none')
mn = min(Y_te_tk[:, 0].min(), Y_pred_tk[:, 0].min())
mx = max(Y_te_tk[:, 0].max(), Y_pred_tk[:, 0].max())
ax.plot([mn, mx], [mn, mx], '--', color=CLR_TEXT, lw=1, alpha=0.6)
ax.text(0.05, 0.92, f'R2 = {metrics["task_cpu_cycles_M"]["R2"]:.4f}',
        transform=ax.transAxes, color=CLR_TEXT, fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8,
                  edgecolor=CLR_SPINE))
ax.set_xlabel('Actual cpu_cycles_M', color=CLR_TEXT, fontsize=9)
ax.set_ylabel('Predicted cpu_cycles_M', color=CLR_TEXT, fontsize=9)


# ── Row 5: Traffic Prediction - Actual vs Predicted ──────────────────
sample_tr = np.random.RandomState(42).choice(len(Y_te_trf),
            min(500, len(Y_te_trf)), replace=False)

# 5a: total_vehicles_present
ax = fig.add_subplot(gs[5, 0])
style_ax(ax, 'Traffic Prediction\nActual vs Predicted (Total Vehicles)')
ax.scatter(Y_te_trf[sample_tr, 0], Y_pred_trf[sample_tr, 0],
           alpha=0.55, s=18, color=CLR_ACCENT1, edgecolors='white',
           linewidth=0.3)
mn = min(Y_te_trf[:, 0].min(), Y_pred_trf[:, 0].min())
mx = max(Y_te_trf[:, 0].max(), Y_pred_trf[:, 0].max())
ax.plot([mn, mx], [mn, mx], '--', color=CLR_BASELINE, lw=1.2, alpha=0.8)
ax.text(0.05, 0.92, f'R2 = {metrics["traffic_total_vehicles_present"]["R2"]:.4f}',
        transform=ax.transAxes, color=CLR_TEXT, fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8,
                  edgecolor=CLR_SPINE))
ax.set_xlabel('Actual Total Vehicles', color=CLR_TEXT, fontsize=9)
ax.set_ylabel('Predicted Total Vehicles', color=CLR_TEXT, fontsize=9)

# 5b: moving_vehicles
ax = fig.add_subplot(gs[5, 1])
style_ax(ax, 'Traffic Prediction\nActual vs Predicted (Moving Vehicles)')
ax.scatter(Y_te_trf[sample_tr, 1], Y_pred_trf[sample_tr, 1],
           alpha=0.55, s=18, color='#6A5B7B', edgecolors='white',
           linewidth=0.3)
mn = min(Y_te_trf[:, 1].min(), Y_pred_trf[:, 1].min())
mx = max(Y_te_trf[:, 1].max(), Y_pred_trf[:, 1].max())
ax.plot([mn, mx], [mn, mx], '--', color=CLR_BASELINE, lw=1.2, alpha=0.8)
ax.text(0.05, 0.92, f'R2 = {metrics["traffic_moving_vehicles"]["R2"]:.4f}',
        transform=ax.transAxes, color=CLR_TEXT, fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8,
                  edgecolor=CLR_SPINE))
ax.set_xlabel('Actual Moving Vehicles', color=CLR_TEXT, fontsize=9)
ax.set_ylabel('Predicted Moving Vehicles', color=CLR_TEXT, fontsize=9)

# 5c: Deadline met rate comparison (bar)
ax = fig.add_subplot(gs[5, 2])
style_ax(ax, 'Deadline Met Rate (%)\nby Task Type')
dl_base_vals = [task_type_stats[t]['baseline_dl_rate'] for t in unique_types]
dl_opt_vals  = [task_type_stats[t]['opt_dl_rate'] for t in unique_types]
ax.bar(x_pos - bar_w/2, dl_base_vals, bar_w, label='Baseline (No-ML)',
       color=CLR_BASELINE, alpha=0.90, edgecolor='white', lw=0.8)
ax.bar(x_pos + bar_w/2, dl_opt_vals, bar_w, label='ML-Optimised',
       color=CLR_OPTIMISED, alpha=0.90, edgecolor='white', lw=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([t.replace('_', '\n') for t in unique_types],
                   fontsize=7.5, color=CLR_TEXT)
ax.set_ylabel('Deadline Met (%)', color=CLR_TEXT, fontsize=9)
ax.set_ylim(0, 112)
style_legend(ax)


# ── Row 6: Summary Table + Research Contributions ────────────────────
ax = fig.add_subplot(gs[6, :2])
style_ax(ax, 'Optimization Results Summary')
ax.axis('off')
ax.grid(False)

table_data = [
    ['Total Cost (USD)',
     f'${baseline_total_cost:.4f}',
     f'${optimised_total_cost:.4f}',
     f'{cost_savings_pct:+.2f}%'],
    ['Avg Cost/Task (USD)',
     f'${base_cost.mean():.6f}',
     f'${opt_cost.mean():.6f}',
     f'{cost_savings_pct:+.2f}%'],
    ['Avg Latency (ms)',
     f'{baseline_avg_lat:.2f}',
     f'{optimised_avg_lat:.2f}',
     f'{latency_change_pct:+.2f}%'],
    ['Median Latency (ms)',
     f'{np.median(base_latency):.2f}',
     f'{np.median(opt_latency):.2f}',
     f'{(1 - np.median(opt_latency)/np.median(base_latency))*100:+.2f}%'],
    ['Deadline Met Rate (%)',
     f'{baseline_dl_rate:.2f}%',
     f'{optimised_dl_rate:.2f}%',
     f'{optimised_dl_rate - baseline_dl_rate:+.2f}pp'],
    ['Total Tasks',
     f'{n_tasks:,}',
     f'{n_tasks:,}',
     '---'],
]
tbl = ax.table(cellText=table_data,
               colLabels=['Metric', 'Baseline (No-ML)', 'ML-Optimised', 'Improvement'],
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 2.0)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor(CLR_ACCENT1)
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#F5F5F5' if r % 2 == 1 else 'white')
        is_imp = (c == 3 and '+' in str(table_data[r-1][3])
                  and table_data[r-1][3] != '---')
        color = '#2E7D32' if is_imp else CLR_TEXT
        cell.set_text_props(color=color, fontweight='bold' if is_imp else 'normal')
    cell.set_edgecolor(CLR_GRIDLINE)

# Research contributions
ax = fig.add_subplot(gs[6, 2])
style_ax(ax, 'Research Contributions')
ax.axis('off')
ax.grid(False)
contributions = [
    "* Winning ML models\n   from Voting Algorithm",
    "* Multi-stage pipeline:\n   Traffic > Vehicles > Tasks",
    "* Signal/Speed-aware\n   bandwidth optimization",
    "* RSU capacity-aware\n   dynamic offloading",
    f"* Cost savings:\n   {cost_savings_pct:+.1f}%",
    f"* Latency improvement:\n   {latency_change_pct:+.1f}%",
    f"* Deadline met rate:\n   {optimised_dl_rate:.1f}%",
]
for i, text in enumerate(contributions):
    ax.text(0.05, 0.95 - i * 0.135, text, transform=ax.transAxes,
            color=CLR_ACCENT1, fontsize=9, fontweight='bold', family='serif',
            verticalalignment='top')

fig.suptitle(
    'ML-Driven Multi-Objective Cost & Latency Optimization\n'
    '2x2 Grid | 4 Regions | 4 RSUs',
    color=CLR_TEXT, fontsize=16, fontweight='bold', y=0.997)

plt.savefig(OUTPUT_MAIN, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n  >> Main visualisation saved -> {OUTPUT_MAIN}")


# =====================================================================
# PHASE 6: COST & LATENCY TREND OVER TIME
# =====================================================================
print("\n" + "-" * 72)
print("  PHASE 6 | Cost & Latency Trends Over Time")
print("-" * 72)

unique_ts_test = sorted(np.unique(ts_indices))

base_ts_cost, opt_ts_cost     = [], []
base_ts_lat,  opt_ts_lat      = [], []
base_ts_dlrate, opt_ts_dlrate = [], []
for ts in unique_ts_test:
    mask = ts_indices == ts
    base_ts_cost.append(base_cost[mask].sum())
    opt_ts_cost.append(opt_cost[mask].sum())
    base_ts_lat.append(base_latency[mask].mean())
    opt_ts_lat.append(opt_latency[mask].mean())
    base_ts_dlrate.append(base_deadline_met[mask].mean() * 100)
    opt_ts_dlrate.append(opt_deadline_met[mask].mean() * 100)

fig2, (ax_c, ax_l, ax_d) = plt.subplots(3, 1, figsize=(16, 13))
fig2.patch.set_facecolor('white')

for ax_trend in [ax_c, ax_l, ax_d]:
    ax_trend.set_facecolor('#FAFAFA')
    ax_trend.grid(True, linestyle='--', linewidth=0.4, color=CLR_GRIDLINE, alpha=0.7)
    ax_trend.tick_params(colors=CLR_TEXT, labelsize=9)
    for spine in ax_trend.spines.values():
        spine.set_edgecolor(CLR_SPINE)
        spine.set_linewidth(0.6)

# Cost trend
ax_c.plot(unique_ts_test, base_ts_cost, color=CLR_BASELINE, lw=2, alpha=0.9,
          label='Baseline Cost', marker='o', markersize=3)
ax_c.plot(unique_ts_test, opt_ts_cost, color=CLR_OPTIMISED, lw=2, alpha=0.9,
          label='ML-Optimised Cost', marker='s', markersize=3)
ax_c.fill_between(unique_ts_test, opt_ts_cost, base_ts_cost,
                  where=[b > o for b, o in zip(base_ts_cost, opt_ts_cost)],
                  alpha=0.15, color=CLR_OPTIMISED, label='Savings Region')
ax_c.set_xlabel('Timestamp Index', color=CLR_TEXT, fontsize=10)
ax_c.set_ylabel('Total Cost (USD)', color=CLR_TEXT, fontsize=10)
ax_c.set_title('Cost per Timestamp: Baseline vs ML-Optimised',
               color=CLR_TEXT, fontsize=13, fontweight='bold')
ax_c.legend(facecolor='white', edgecolor=CLR_SPINE, labelcolor=CLR_TEXT, fontsize=9)

# Latency trend
ax_l.plot(unique_ts_test, base_ts_lat, color=CLR_BASELINE, lw=2, alpha=0.9,
          label='Baseline Latency', marker='o', markersize=3)
ax_l.plot(unique_ts_test, opt_ts_lat, color=CLR_OPTIMISED, lw=2, alpha=0.9,
          label='ML-Optimised Latency', marker='s', markersize=3)
ax_l.fill_between(unique_ts_test, opt_ts_lat, base_ts_lat,
                  alpha=0.12, color=CLR_OPTIMISED, label='Improvement Region')
ax_l.set_xlabel('Timestamp Index', color=CLR_TEXT, fontsize=10)
ax_l.set_ylabel('Avg Latency (ms)', color=CLR_TEXT, fontsize=10)
ax_l.set_title('Avg Latency per Timestamp: Baseline vs ML-Optimised',
               color=CLR_TEXT, fontsize=13, fontweight='bold')
ax_l.legend(facecolor='white', edgecolor=CLR_SPINE, labelcolor=CLR_TEXT, fontsize=9)

# Deadline met rate trend
ax_d.plot(unique_ts_test, base_ts_dlrate, color=CLR_BASELINE, lw=2, alpha=0.9,
          label='Baseline Deadline %', marker='o', markersize=3)
ax_d.plot(unique_ts_test, opt_ts_dlrate, color=CLR_OPTIMISED, lw=2, alpha=0.9,
          label='ML-Optimised Deadline %', marker='s', markersize=3)
ax_d.set_xlabel('Timestamp Index', color=CLR_TEXT, fontsize=10)
ax_d.set_ylabel('Deadline Met (%)', color=CLR_TEXT, fontsize=10)
ax_d.set_title('Deadline Met Rate per Timestamp: Baseline vs ML-Optimised',
               color=CLR_TEXT, fontsize=13, fontweight='bold')
ax_d.legend(facecolor='white', edgecolor=CLR_SPINE, labelcolor=CLR_TEXT, fontsize=9)

plt.tight_layout(pad=2.5)
plt.savefig(OUTPUT_TREND, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  >> Trend visualisation saved -> {OUTPUT_TREND}")


# =====================================================================
# FINAL SUMMARY
# =====================================================================
print("\n" + "=" * 72)
print("  SIMULATION COMPLETE")
print("=" * 72)
print(f"""
  ML Models Used (from Voting Algorithm):
    Traffic:  Linear Regression   (R2={metrics['traffic_total_vehicles_present']['R2']:.4f}, {metrics['traffic_moving_vehicles']['R2']:.4f})
    Vehicle:  Decision Tree (speed R2={metrics['vehicle_speed']['R2']:.4f}, type Acc={metrics['vehicle_type']['Accuracy']:.4f})
              Linear Regression (signal R2={metrics['vehicle_signal']['R2']:.4f})
    Task:     Random Forest (cpu R2={metrics['task_cpu_cycles_M']['R2']:.4f}, data R2={metrics['task_data_size_KB']['R2']:.4f}, deadline R2={metrics['task_deadline_ms']['R2']:.4f})

  Grid: {GRID_ROWS}x{GRID_COLS} = {NUM_REGIONS} regions, {NUM_REGIONS} RSUs
  Test set: {n_tasks:,} tasks over {len(unique_ts_test)} timestamps

  Results:
    Cost savings:          {cost_savings_pct:+.2f}%
    Latency improvement:   {latency_change_pct:+.2f}%
    Deadline met (base):   {baseline_dl_rate:.2f}%
    Deadline met (opt):    {optimised_dl_rate:.2f}%

  Output files:
    {OUTPUT_MAIN}
    {OUTPUT_TREND}
""")
print("=" * 72)


# =====================================================================
# PHASE 7: COST COMPARISON — Ours vs Paper 2 (NSGA-II)
# =====================================================================
print("\n" + "-" * 72)
print("  PHASE 7 | Cost Comparison — Ours vs Paper 2 (NSGA-II)")
print("-" * 72)

OUTPUT_PAPERS = "cost_comparison_papers.png"

# ── Simulate Paper 2's NSGA-II strategy on our test set ──────────────
# Paper 2 (NSGA-II 2021): Multi-objective cost+latency but STATIC
#   - Equal weights (w1=w2=0.5) for ALL task types
#   - No ML prediction (uses actual values)
#   - No signal/speed-aware bandwidth scaling
#   - No RSU capacity awareness
#
# Ours: ML-driven, task-type-aware dynamic weights, signal/speed
#       bandwidth scaling, RSU capacity awareness.

print("  Simulating Paper 2 strategy (NSGA-II, equal w1=w2=0.5, static) ...")
p2_cost = np.zeros(n_tasks)
for i in range(n_tasks):
    best_F = float('inf')
    chosen_cost = 0.0
    for infra_name, infra in INFRA.items():
        lat = compute_latency(actual_cpu[i], actual_data[i], infra,
                              -65.0, 40.0, infra_name)   # no signal/speed awareness
        cst = compute_cost(actual_cpu[i], actual_data[i], infra)
        norm_lat = lat / max_lat if max_lat > 0 else 0
        norm_cst = cst / max_cst if max_cst > 0 else 0
        F = 0.5 * norm_lat + 0.5 * norm_cst               # equal weights, static
        if F < best_F:
            best_F = F
            chosen_cost = cst
    p2_cost[i] = chosen_cost

# ── Build per-timestamp cost series for Paper 2 ──
p2_ts_cost = []
for ts in unique_ts_test:
    mask = ts_indices == ts
    p2_ts_cost.append(p2_cost[mask].sum())

p2_total  = p2_cost.sum()
our_total = optimised_total_cost
savings   = (1 - our_total / p2_total) * 100 if p2_total > 0 else 0

print(f"\n  Paper 2 (NSGA-II) total cost : ${p2_total:.4f}")
print(f"  Ours (ML-Optimised) total   : ${our_total:.4f}")
print(f"  Our savings vs Paper 2      : {savings:+.2f}%")


# ── Plot: Grouped Bar Chart — Baseline vs Paper 2 vs Ours per timestamp ──
fig3, ax_cmp = plt.subplots(figsize=(18, 6))
fig3.patch.set_facecolor('white')

ax_cmp.set_facecolor('#FAFAFA')
ax_cmp.grid(True, axis='y', linestyle='--', linewidth=0.4, color=CLR_GRIDLINE, alpha=0.7)
for spine in ax_cmp.spines.values():
    spine.set_edgecolor(CLR_SPINE)
    spine.set_linewidth(0.6)

x = np.arange(len(unique_ts_test))
width = 0.28   # width of each bar

# Baseline
ax_cmp.bar(x - width, base_ts_cost, width, color='#7FB3D3', alpha=0.9,
           label='Baseline (No-ML Heuristic)', edgecolor='white', lw=0.8)
# Paper 2
ax_cmp.bar(x, p2_ts_cost, width, color=CLR_BASELINE, alpha=0.9,
           label='Paper 2 — NSGA-II (static, equal weights)', edgecolor='white', lw=0.8)
# Ours
ax_cmp.bar(x + width, opt_ts_cost, width, color=CLR_OPTIMISED, alpha=0.9,
           label='Ours — ML-Driven Multi-Objective', edgecolor='white', lw=0.8)

ax_cmp.set_xlabel('Timestamp Index', color=CLR_TEXT, fontsize=10, fontweight='bold')
ax_cmp.set_ylabel('Total Cost (USD)', color=CLR_TEXT, fontsize=10, fontweight='bold')
ax_cmp.set_title(
    'Cost per Timestamp: Baseline vs Paper 2 (NSGA-II) vs Ours',
    color=CLR_TEXT, fontsize=14, fontweight='bold', pad=15)
ax_cmp.set_xticks(x)
ax_cmp.set_xticklabels(unique_ts_test, fontsize=9, rotation=45 if len(unique_ts_test)>15 else 0)
ax_cmp.tick_params(colors=CLR_TEXT, labelsize=9)
ax_cmp.legend(facecolor='white', edgecolor=CLR_SPINE, labelcolor=CLR_TEXT,
              fontsize=10, loc='upper right')

plt.tight_layout(pad=2.0)
plt.savefig(OUTPUT_PAPERS, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n  >> Paper comparison chart saved -> {OUTPUT_PAPERS}")
print("=" * 72)
