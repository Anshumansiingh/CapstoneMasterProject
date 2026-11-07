"""
IPhSO - Step 3: LightGBM (Poisson) demand forecasting model
===========================================================
Run AFTER step 1 and step 2, from the same folder.
Reads  : processed/hourly_demand.csv, processed/baseline_forecasts.csv,
         processed/baseline_results.csv
Writes : processed/model_comparison.csv
         processed/model_mae_by_position.csv, processed/model_mae_by_day.csv
         processed/test_forecasts_all_models.csv
         processed/figures/07_feature_importance.png
         processed/figures/08_test_week_models.png

Install once:  pip install lightgbm
(If LightGBM is missing, scikit-learn's HistGradientBoostingRegressor, the
 same gradient-boosting method, is used instead so the script still runs.)

Two model variants are compared:
  A) calendar-only : hour, weekday, weekend, holiday, position
                     -> learns the typical weekly profile from all 3 train weeks
  B) calendar + lag: also "orders in the same slot one week earlier"
                     -> can react to recent changes, but loses week 1 for
                        training because week 1 has no previous week
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

try:
    from lightgbm import LGBMRegressor
    ENGINE = "lightgbm"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    ENGINE = "sklearn-hgb"
print(f"Gradient boosting engine: {ENGINE}")

PROC = Path("processed")
FIG = PROC / "figures"
SEED = 42

# ---------------------------------------------------------------------------
# 1. BUILD THE MODELLING TABLE (same position x hour level as step 2)
# ---------------------------------------------------------------------------
raw = pd.read_csv(PROC / "hourly_demand.csv", parse_dates=["demand_hour"])
raw = raw.dropna(subset=["position"])
START, END = pd.Timestamp("2024-05-10"), pd.Timestamp("2024-06-06 23:00")
raw = raw[(raw.demand_hour >= START) & (raw.demand_hour <= END)]

df = (raw.groupby(["demand_hour", "position"], as_index=False)
         .agg(n_orders=("n_orders", "sum"), is_holiday=("is_holiday", "max")))
df["hour"] = df.demand_hour.dt.hour
df["dow"] = df.demand_hour.dt.dayofweek
df["is_weekend"] = (df.dow >= 5).astype(int)
df["is_holiday"] = df.is_holiday.astype(int)
df["position"] = df.position.astype("category")

# Lag feature: same position, same hour, 7 days earlier.
# Valid for a Friday forecast because the test week starts on a Friday:
# every test-day lag falls in the previous (already observed) week.
df = df.sort_values(["position", "demand_hour"])
s = df.set_index(["position", "demand_hour"]).n_orders
df["lag_1wk"] = s.reindex(list(zip(df.position,
                    df.demand_hour - pd.Timedelta(weeks=1)))).values

TEST_START = pd.Timestamp("2024-05-31")   # week 4 = test (same as step 2)
VAL_START = pd.Timestamp("2024-05-24")    # week 3 = validation for tuning

FEATS_A = ["hour", "dow", "is_weekend", "is_holiday", "position"]
FEATS_B = FEATS_A + ["lag_1wk"]


def make_model(n_iter):
    """Small, regularised trees: we only have ~5,500 training rows."""
    if ENGINE == "lightgbm":
        return LGBMRegressor(objective="poisson", n_estimators=n_iter,
                             learning_rate=0.05, num_leaves=15,
                             min_child_samples=20, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.9,
                             random_state=SEED, verbose=-1)
    return HistGradientBoostingRegressor(loss="poisson", max_iter=n_iter,
                                         learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=20,
                                         categorical_features="from_dtype",
                                         random_state=SEED)


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


# ---------------------------------------------------------------------------
# 2. TRAIN EACH VARIANT
#    a) tune number of trees: fit on early train weeks, score on week 3
#    b) refit on ALL train weeks with the chosen number of trees
#    c) predict the untouched test week
# ---------------------------------------------------------------------------
GRID = [50, 100, 200, 300, 500, 800]
test = df[df.demand_hour >= TEST_START].copy()
fitted = {}

for name, feats in [("lgbm_calendar", FEATS_A), ("lgbm_calendar_lag", FEATS_B)]:
    train = df[df.demand_hour < TEST_START].dropna(subset=feats)
    fit_part = train[train.demand_hour < VAL_START]
    val_part = train[train.demand_hour >= VAL_START]

    val_scores = {}
    for n in GRID:
        m = make_model(n).fit(fit_part[feats], fit_part.n_orders)
        val_scores[n] = mae(val_part.n_orders, m.predict(val_part[feats]))
    best_n = min(val_scores, key=val_scores.get)
    print(f"\n{name}: train rows={len(train)}, validation MAE by trees="
          f"{ {k: round(v, 3) for k, v in val_scores.items()} } -> {best_n}")

    model = make_model(best_n).fit(train[feats], train.n_orders)
    test[f"pred_{name}"] = model.predict(test[feats])
    fitted[name] = (model, feats)

# ---------------------------------------------------------------------------
# 3. COMPARE WITH THE BASELINES FROM STEP 2 (identical test rows)
# ---------------------------------------------------------------------------
base = pd.read_csv(PROC / "baseline_forecasts.csv", parse_dates=["demand_hour"])
base_cols = [c for c in base.columns if c.startswith("pred_")]
test["position"] = test.position.astype(str)
test = test.merge(base[["demand_hour", "position"] + base_cols],
                  on=["demand_hour", "position"], how="left")

MODELS = base_cols + ["pred_lgbm_calendar", "pred_lgbm_calendar_lag"]
rows = []
for m in MODELS:
    err = test[m] - test.n_orders
    rows.append({"model": m.replace("pred_", ""), "MAE": err.abs().mean(),
                 "RMSE": np.sqrt((err ** 2).mean()),
                 "bias_total_orders": err.sum()})
comp = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
bench = comp.loc[comp.model == "avg_3wk", "MAE"].iloc[0]
comp["improvement_vs_avg_3wk_%"] = (bench - comp.MAE) / bench * 100
print("\n=== Test-week comparison (orders per position-hour) ===")
print(comp.round(3).to_string(index=False))

lbl = lambda m: m.replace("pred_", "")
by_pos = test.groupby("position").apply(
    lambda g: pd.Series({lbl(m): mae(g.n_orders, g[m]) for m in MODELS}))
by_pos["mean_orders_per_hour"] = test.groupby("position").n_orders.mean()
by_day = test.assign(day=test.demand_hour.dt.strftime("%a %d %b")).groupby(
    "day", sort=False).apply(
    lambda g: pd.Series({lbl(m): mae(g.n_orders, g[m]) for m in MODELS}))
print("\nMAE by position:"); print(by_pos.round(3).to_string())
print("\nMAE by test day:"); print(by_day.round(3).to_string())

# Peak-hour check: the 7-9 AM rounds matter most for staffing
peak = test[test.demand_hour.dt.hour.between(7, 9)]
print("\nMAE in 07:00-09:59 peak only:")
print({lbl(m): round(mae(peak.n_orders, peak[m]), 3) for m in MODELS})

# ---------------------------------------------------------------------------
# 4. FEATURE IMPORTANCE (permutation: how much worse is MAE if a feature
#    is shuffled?) - explains WHAT the model relies on, for Sec 1.8
# ---------------------------------------------------------------------------
best_lgbm = min(["lgbm_calendar", "lgbm_calendar_lag"],
                key=lambda n: comp.set_index("model").MAE[n])
model, feats = fitted[best_lgbm]
X_imp = df[df.demand_hour >= TEST_START][feats]
y_imp = df[df.demand_hour >= TEST_START].n_orders
imp = permutation_importance(model, X_imp, y_imp, n_repeats=5,
                             random_state=SEED,
                             scoring="neg_mean_absolute_error")
imp_s = pd.Series(imp.importances_mean, index=feats).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp_s.index, imp_s.values)
ax.set(title=f"Feature importance - {best_lgbm}",
       xlabel="Increase in MAE when feature is shuffled")
fig.tight_layout(); fig.savefig(FIG / "07_feature_importance.png", dpi=120)
plt.close(fig)
print("\nPermutation importance:", imp_s.round(3).to_dict())

# ---------------------------------------------------------------------------
# 5. PLOT: actual vs baseline vs best LightGBM for the busiest position
# ---------------------------------------------------------------------------
busiest = test.groupby("position").n_orders.sum().idxmax()
t = test[test.position == busiest]
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t.demand_hour, t.n_orders, label="Actual", lw=1.6, color="black")
ax.plot(t.demand_hour, t.pred_avg_3wk, label="avg_3wk", lw=1.2, alpha=0.8)
ax.plot(t.demand_hour, t[f"pred_{best_lgbm}"], label=best_lgbm, lw=1.2, alpha=0.8)
ax.set(title=f"Test week - {busiest}", ylabel="Orders per hour")
ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(FIG / "08_test_week_models.png", dpi=120); plt.close(fig)

comp.to_csv(PROC / "model_comparison.csv", index=False)
by_pos.to_csv(PROC / "model_mae_by_position.csv")
by_day.to_csv(PROC / "model_mae_by_day.csv")
test.to_csv(PROC / "test_forecasts_all_models.csv", index=False)
print("\nDone. Results in processed/, figures in processed/figures/")