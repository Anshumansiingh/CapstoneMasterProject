"""
IPhSO - Step 2: Exploratory analysis, train/test split, baseline forecasts
==========================================================================
Run AFTER iphso_step1_data_prep.py, from the same folder.
Reads  : processed/hourly_demand.csv
Writes : processed/figures/*.png          (exploratory plots)
         processed/baseline_results.csv   (MAE of each baseline)
         processed/baseline_forecasts.csv (test-week predictions vs actuals)

Forecast level: orders per MLA POSITION per HOUR (order types are added up).
Reason: at position x hour x order type the table is ~94% zeros, too sparse
to forecast directly. Order-type shares can be applied afterwards.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC = Path("processed")
FIG = PROC / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ---------------------------------------------------------------------------
# 1. LOAD AND AGGREGATE TO POSITION x HOUR
# ---------------------------------------------------------------------------
raw = pd.read_csv(PROC / "hourly_demand.csv", parse_dates=["demand_hour"])
raw = raw.dropna(subset=["position"])            # drop unmapped locations

START, END = pd.Timestamp("2024-05-10"), pd.Timestamp("2024-06-06 23:00")
raw = raw[(raw.demand_hour >= START) & (raw.demand_hour <= END)]  # 28 full days

df = (raw.groupby(["demand_hour", "position"], as_index=False)
         [["n_orders", "workload_hours"]].sum())
df["hour"] = df.demand_hour.dt.hour
df["dow"] = df.demand_hour.dt.dayofweek
df["date"] = df.demand_hour.dt.normalize()
print(f"Position-hour table: {df.shape}, positions: {df.position.nunique()}")

# ---------------------------------------------------------------------------
# 2. EXPLORATORY PLOTS  (what patterns should a model learn?)
# ---------------------------------------------------------------------------
# 2a. Daily totals over time: trend? weekly cycle? holiday dip?
daily = df.groupby("date").n_orders.sum()
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(daily.index, daily.values, marker="o")
for d in daily.index[daily.index.dayofweek >= 5]:
    ax.axvspan(d, d + pd.Timedelta(days=1), color="grey", alpha=0.15)
ax.axvline(pd.Timestamp("2024-05-20"), color="red", ls="--", label="Victoria Day")
ax.set(title="Daily inpatient draws (weekends shaded)", ylabel="Orders")
ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(FIG / "01_daily_totals.png", dpi=120); plt.close(fig)

# 2b. Average orders by hour of day: weekday vs weekend
hp = (df.groupby(["date", "hour"]).n_orders.sum().reset_index()
        .assign(weekend=lambda x: x.date.dt.dayofweek >= 5)
        .groupby(["weekend", "hour"]).n_orders.mean().unstack(0))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hp.index, hp[False], marker="o", label="Weekday")
ax.plot(hp.index, hp[True], marker="o", label="Weekend")
ax.set(title="Average orders per hour of day (all positions)",
       xlabel="Hour", ylabel="Orders", xticks=range(24))
ax.legend(); fig.tight_layout()
fig.savefig(FIG / "02_hour_of_day.png", dpi=120); plt.close(fig)

# 2c. Heatmap day-of-week x hour
heat = (df.groupby(["date", "dow", "hour"]).n_orders.sum().reset_index()
          .groupby(["dow", "hour"]).n_orders.mean().unstack())
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd")
ax.set(yticks=range(7), yticklabels=DAYS, xticks=range(24),
       title="Average orders: day of week x hour", xlabel="Hour")
fig.colorbar(im, ax=ax, label="Orders"); fig.tight_layout()
fig.savefig(FIG / "03_heatmap_dow_hour.png", dpi=120); plt.close(fig)

# 2d. Total volume per position
pos_tot = df.groupby("position").n_orders.sum().sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(pos_tot.index, pos_tot.values)
ax.set(title="Total orders by MLA position (28 days)", xlabel="Orders")
fig.tight_layout(); fig.savefig(FIG / "04_orders_by_position.png", dpi=120); plt.close(fig)

# 2e. Hourly profile per position (do positions peak at different times?)
pp = df.groupby(["position", "hour"]).n_orders.mean().unstack(0)
fig, ax = plt.subplots(figsize=(11, 5))
for c in pp.columns:
    ax.plot(pp.index, pp[c], label=c)
ax.set(title="Average orders per hour, by position", xlabel="Hour",
       ylabel="Orders", xticks=range(24))
ax.legend(ncol=3, fontsize=8); fig.tight_layout()
fig.savefig(FIG / "05_hour_profile_by_position.png", dpi=120); plt.close(fig)

# 2f. How sparse is the target? (share of zero hours per position)
zero_share = df.assign(z=df.n_orders == 0).groupby("position").z.mean().sort_values()
print("\nShare of hours with zero orders, by position:")
print(zero_share.round(2).to_string())

# ---------------------------------------------------------------------------
# 3. TIME-BASED TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
# Never shuffle time series: train on the past, test on the future.
# 28 days = 4 weeks. Weeks 1-3 train, week 4 test (May 31 - Jun 6).
TEST_START = pd.Timestamp("2024-05-31")
train = df[df.demand_hour < TEST_START]
test = df[df.demand_hour >= TEST_START].copy()
print(f"\nTrain: {train.date.min().date()} to {train.date.max().date()} ({len(train)} rows)")
print(f"Test : {test.date.min().date()} to {test.date.max().date()} ({len(test)} rows)")

# ---------------------------------------------------------------------------
# 4. BASELINE FORECASTS
# ---------------------------------------------------------------------------
# Lookup: same position, same weekday, same hour, k weeks earlier.
hist = df.set_index(["position", "demand_hour"]).n_orders

def lag_weeks(frame, k):
    idx = list(zip(frame.position, frame.demand_hour - pd.Timedelta(weeks=k)))
    return hist.reindex(idx).values

# Baseline 1: seasonal naive = value one week earlier
test["pred_seasonal_naive"] = lag_weeks(test, 1)
# Baseline 2: average of the same slot over the previous 3 weeks
test["pred_avg_3wk"] = np.nanmean([lag_weeks(test, k) for k in (1, 2, 3)], axis=0)
# Baseline 3: weighted average, recent weeks count more (weights 3:2:1)
w = np.array([3, 2, 1])[:, None]
lags = np.array([lag_weeks(test, k) for k in (1, 2, 3)])
test["pred_wavg_3wk"] = (lags * w).sum(0) / w.sum()

# ---------------------------------------------------------------------------
# 5. EVALUATE WITH MAE (+ RMSE and total bias)
# ---------------------------------------------------------------------------
MODELS = ["pred_seasonal_naive", "pred_avg_3wk", "pred_wavg_3wk"]
rows = []
for m in MODELS:
    err = test[m] - test.n_orders
    rows.append({"model": m.replace("pred_", ""),
                 "MAE": err.abs().mean(),
                 "RMSE": np.sqrt((err ** 2).mean()),
                 "bias_total_orders": err.sum()})       # + = over-forecast
results = pd.DataFrame(rows).sort_values("MAE")
print("\nTest-week accuracy (orders per position-hour):")
print(results.round(3).to_string(index=False))

# MAE by test day (lead time) - required by proposal Sec 2.6.1
by_day = (test.assign(day=test.demand_hour.dt.strftime("%a %d %b"))
              .groupby("day", sort=False)
              .apply(lambda g: pd.Series({m.replace("pred_", ""):
                     (g[m] - g.n_orders).abs().mean() for m in MODELS})))
print("\nMAE by test day:"); print(by_day.round(3).to_string())

# MAE by position - where is forecasting hardest?
by_pos = (test.groupby("position")
              .apply(lambda g: pd.Series({m.replace("pred_", ""):
                     (g[m] - g.n_orders).abs().mean() for m in MODELS})))
by_pos["mean_orders_per_hour"] = test.groupby("position").n_orders.mean()
print("\nMAE by position:"); print(by_pos.round(3).to_string())

# Plot: actual vs best baseline for the busiest position
best = "pred_" + results.iloc[0].model
busiest = pos_tot.index[-1]
t = test[test.position == busiest]
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t.demand_hour, t.n_orders, label="Actual", lw=1.5)
ax.plot(t.demand_hour, t[best], label=best.replace("pred_", ""), lw=1.5, alpha=0.8)
ax.set(title=f"Test week: actual vs {best.replace('pred_', '')} - {busiest}",
       ylabel="Orders per hour")
ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(FIG / "06_test_week_actual_vs_baseline.png", dpi=120); plt.close(fig)

results.to_csv(PROC / "baseline_results.csv", index=False)
by_day.to_csv(PROC / "baseline_mae_by_day.csv")
by_pos.to_csv(PROC / "baseline_mae_by_position.csv")
test.to_csv(PROC / "baseline_forecasts.csv", index=False)
print("\nDone. Figures in processed/figures/, results in processed/")