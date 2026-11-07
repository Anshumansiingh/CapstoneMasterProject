"""
IPhSO - Step 4: Weekly production forecast + workload
=====================================================
This is the "Friday afternoon run" from the proposal.
Run AFTER steps 1-3, from the same folder.

Reads  : processed/hourly_demand.csv, processed/orders_clean.csv,
         static data/parameters.csv
Writes : processed/forecast_next_week.csv       (position x hour x order type)
         processed/workload_next_week.csv       (position x hour  -> optimizer input)
         processed/figures/09_forecast_next_week.png
         processed/figures/10_mlas_needed_heatmap.png

What it does:
  1. Retrain the selected model (LightGBM Poisson, calendar features,
     500 trees) on ALL available history.
  2. Forecast orders for next Monday-Sunday.
  3. Split orders into order types using smoothed historical shares.
  4. Forecast patients using historical patients-per-order ratios.
  5. Convert orders -> workload hours -> MLAs needed.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor

PROC, FIG = Path("processed"), Path("processed") / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
N_TREES = 500                      # chosen by validation in step 3
FEATS = ["hour", "dow", "is_weekend", "is_holiday", "position"]

# BC statutory holidays (extend when forecasting other periods)
HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2024-02-19", "2024-03-29", "2024-05-20", "2024-07-01",
    "2024-08-05", "2024-09-02", "2024-09-30", "2024-10-14", "2024-11-11",
    "2024-12-25"])

params = pd.read_csv("static data/parameters.csv", header=None, index_col=0)[1]
CAPACITY = float(params["mla_hourly_workload_capacity"])   # CONFIRM meaning
print(f"MLA hourly workload capacity from parameters.csv: {CAPACITY}")


def add_calendar(frame):
    frame["hour"] = frame.demand_hour.dt.hour
    frame["dow"] = frame.demand_hour.dt.dayofweek
    frame["is_weekend"] = (frame.dow >= 5).astype(int)
    frame["is_holiday"] = frame.demand_hour.dt.normalize().isin(HOLIDAYS).astype(int)
    return frame


# ---------------------------------------------------------------------------
# 1. TRAINING DATA = ALL HISTORY (all 4 weeks, test week included)
# ---------------------------------------------------------------------------
raw = pd.read_csv(PROC / "hourly_demand.csv", parse_dates=["demand_hour"])
raw = raw.dropna(subset=["position"])
START, END = pd.Timestamp("2024-05-10"), pd.Timestamp("2024-06-06 23:00")
raw = raw[(raw.demand_hour >= START) & (raw.demand_hour <= END)]

hist = (raw.groupby(["demand_hour", "position"], as_index=False)
           .agg(n_orders=("n_orders", "sum")))
hist = add_calendar(hist)
POSITIONS = sorted(hist.position.unique())
hist["position"] = pd.Categorical(hist.position, categories=POSITIONS)
print(f"Training rows: {len(hist)} ({hist.demand_hour.min().date()} to "
      f"{hist.demand_hour.max().date()})")

model = LGBMRegressor(objective="poisson", n_estimators=N_TREES,
                      learning_rate=0.05, num_leaves=15, min_child_samples=20,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                      random_state=SEED, verbose=-1)
model.fit(hist[FEATS], hist.n_orders)

# ---------------------------------------------------------------------------
# 2. FORECAST NEXT MONDAY - SUNDAY
#    Run date = the Friday after the data ends; target = the following week.
# ---------------------------------------------------------------------------
last_day = hist.demand_hour.max().normalize()
run_date = last_day + pd.Timedelta(days=(4 - last_day.dayofweek) % 7 or 7)
week_start = run_date + pd.Timedelta(days=3)                  # Monday
hours = pd.date_range(week_start, week_start + pd.Timedelta(hours=167), freq="h")
print(f"Forecast run (Friday): {run_date.date()}  ->  "
      f"forecast week {week_start.date()} to {hours[-1].date()}")

fc = pd.DataFrame([(h, p) for h in hours for p in POSITIONS],
                  columns=["demand_hour", "position"])
fc = add_calendar(fc)
fc["position"] = pd.Categorical(fc.position, categories=POSITIONS)
fc["pred_orders"] = model.predict(fc[FEATS])
fc["position"] = fc.position.astype(str)
fc["lead_days"] = (fc.demand_hour.dt.normalize() - run_date).dt.days  # 3..9

# ---------------------------------------------------------------------------
# 3. ORDER-TYPE SPLIT  (smoothed historical shares)
#    share(type | position, hour) is noisy when a slot had few orders, so it
#    is blended with the position's overall share:
#    share = (count_slot_type + K * share_position_type) / (count_slot + K)
# ---------------------------------------------------------------------------
K = 10
orders = pd.read_csv(PROC / "orders_clean.csv", parse_dates=["scheduled_date"])
orders = orders.dropna(subset=["position"])
orders = orders[(orders.scheduled_date >= START) & (orders.scheduled_date <= END)]
orders["hour"] = orders.scheduled_date.dt.hour

pos_share = (orders.groupby(["position", "order_type"]).size()
             / orders.groupby("position").size()).rename("pos_share").reset_index()
slot = orders.groupby(["position", "hour", "order_type"]).size().rename("n").reset_index()
slot_tot = orders.groupby(["position", "hour"]).size().rename("n_slot").reset_index()

grid = (pd.DataFrame({"hour": range(24)})
          .merge(pos_share, how="cross")
          .merge(slot, on=["position", "hour", "order_type"], how="left")
          .merge(slot_tot, on=["position", "hour"], how="left")
          .fillna({"n": 0, "n_slot": 0}))
grid["share"] = (grid.n + K * grid.pos_share) / (grid.n_slot + K)
grid["share"] /= grid.groupby(["position", "hour"]).share.transform("sum")

by_type = fc.merge(grid[["position", "hour", "order_type", "share"]],
                   on=["position", "hour"], how="left")
by_type["pred_orders_type"] = by_type.pred_orders * by_type.share

# ---------------------------------------------------------------------------
# 4. PATIENTS  (no patient ID in data -> unique rooms per hour, as in step 1)
# ---------------------------------------------------------------------------
orders["demand_hour"] = orders.scheduled_date.dt.floor("h")
ph = orders.groupby(["position", "demand_hour"]).agg(
    o=("accession", "size"), p=("room", "nunique"))
pat_ratio = (ph.groupby("position").p.sum() / ph.groupby("position").o.sum())
fc["pred_patients"] = fc.pred_orders * fc.position.map(pat_ratio)
print("\nPatients per order, by position:"); print(pat_ratio.round(2).to_string())

# ---------------------------------------------------------------------------
# 5. WORKLOAD HOURS AND MLAs NEEDED
#    minutes per draw = order-weighted historical average for each position
#    (positions contain units with different service times)
# ---------------------------------------------------------------------------
svc_min = orders.groupby("position").service_min.mean()
fc["service_min_per_order"] = fc.position.map(svc_min)
fc["workload_hours"] = fc.pred_orders * fc.service_min_per_order / 60
fc["mlas_needed"] = fc.workload_hours / CAPACITY             # CONFIRM formula
print("\nService minutes per order, by position:"); print(svc_min.round(2).to_string())

# ---------------------------------------------------------------------------
# 6. SAVE + SUMMARY + PLOTS
# ---------------------------------------------------------------------------
fc["run_date"] = run_date.date()
fc["model"] = f"LightGBM-Poisson-calendar-{N_TREES}"
out_cols = ["run_date", "model", "demand_hour", "lead_days", "position",
            "pred_orders", "pred_patients", "service_min_per_order",
            "workload_hours", "mlas_needed"]
fc[out_cols].to_csv(PROC / "workload_next_week.csv", index=False)
by_type[["demand_hour", "position", "order_type", "share",
         "pred_orders_type"]].assign(run_date=run_date.date()).to_csv(
    PROC / "forecast_next_week.csv", index=False)

daily = fc.groupby(fc.demand_hour.dt.date)[
    ["pred_orders", "pred_patients", "workload_hours"]].sum()
print("\nForecast by day:"); print(daily.round(1).to_string())
print(f"\nWeek total: {fc.pred_orders.sum():.0f} orders, "
      f"{fc.workload_hours.sum():.0f} MLA workload hours")
print("\nPeak MLAs needed per position (busiest hour of the week):")
print(fc.groupby("position").mlas_needed.max().round(2).sort_values().to_string())

# Check: forecast type split adds back up to the total
gap = abs(by_type.pred_orders_type.sum() - fc.pred_orders.sum())
print(f"\nOrder-type split check (should be ~0): {gap:.6f}")

# Plot 1: forecast hourly orders for the week, all positions stacked
pv = fc.pivot_table(index="demand_hour", columns="position", values="pred_orders")
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.stackplot(pv.index, pv.T.values, labels=pv.columns)
ax.set(title=f"Forecast orders per hour, week of {week_start.date()}",
       ylabel="Orders")
ax.legend(ncol=6, fontsize=7, loc="upper right"); fig.autofmt_xdate()
fig.tight_layout(); fig.savefig(FIG / "09_forecast_next_week.png", dpi=120)
plt.close(fig)

# Plot 2: average MLAs needed by position x hour
hm = fc.groupby(["position", "hour"]).mlas_needed.mean().unstack()
fig, ax = plt.subplots(figsize=(12, 4.5))
im = ax.imshow(hm.values, aspect="auto", cmap="Blues")
ax.set(yticks=range(len(hm)), yticklabels=hm.index, xticks=range(24),
       xlabel="Hour", title="Average MLAs needed (forecast week)")
fig.colorbar(im, ax=ax, label="MLAs"); fig.tight_layout()
fig.savefig(FIG / "10_mlas_needed_heatmap.png", dpi=120); plt.close(fig)

print("\nDone. Optimizer input: processed/workload_next_week.csv")