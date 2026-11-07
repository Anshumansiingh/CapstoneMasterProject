"""
IPhSO - Step 4b: Forecasts by floor and unit
=============================================
Run AFTER step 4, from the same folder. Does not retrain any model.

Reads  : processed/workload_next_week.csv   (LightGBM forecast per position-hour)
         processed/orders_clean.csv         (history, for unit shares)
Writes : processed/forecast_by_unit_next_week.csv           (unit x hour)
         processed/forecast_by_unit_ordertype_next_week.csv (unit x hour x order type)
         processed/forecast_by_floor_next_week.csv          (floor x hour)
         processed/figures/11_forecast_by_unit.png

Method (top-down allocation, Section 2.2.5 of the report):
  Each position's hourly forecast is divided among its units using the
  unit's historical share of that position's orders in that hour, smoothed
  toward the unit's overall share when the hour has few orders:
      share = (n_unit,hour + K * share_unit) / (n_position,hour + K)
  The same idea then splits each unit's forecast into order types.
  All splits add back exactly to the position forecast.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC, FIG = Path("processed"), Path("processed") / "figures"
FIG.mkdir(parents=True, exist_ok=True)
K = 10
START, END = pd.Timestamp("2024-05-10"), pd.Timestamp("2024-06-06 23:00")

fc = pd.read_csv(PROC / "workload_next_week.csv", parse_dates=["demand_hour"])
fc["hour"] = fc.demand_hour.dt.hour

o = pd.read_csv(PROC / "orders_clean.csv", parse_dates=["scheduled_date"])
o = o.dropna(subset=["position"])
o = o[(o.scheduled_date >= START) & (o.scheduled_date <= END)].copy()
o["hour"] = o.scheduled_date.dt.hour
o["floor"] = o.floor.astype(int)
o["unit"] = o.floor.astype(str) + o.wing.astype(str)


def smoothed_shares(df, group, child, k=K):
    """share(child | group, hour), smoothed toward share(child | group)."""
    overall = (df.groupby(group + [child]).size()
               / df.groupby(group).size()).rename("overall").reset_index()
    slot = df.groupby(group + ["hour", child]).size().rename("n").reset_index()
    tot = df.groupby(group + ["hour"]).size().rename("n_slot").reset_index()
    g = (pd.DataFrame({"hour": range(24)}).merge(overall, how="cross")
           .merge(slot, on=group + ["hour", child], how="left")
           .merge(tot, on=group + ["hour"], how="left")
           .fillna({"n": 0, "n_slot": 0}))
    g["share"] = (g.n + k * g.overall) / (g.n_slot + k)
    g["share"] /= g.groupby(group + ["hour"]).share.transform("sum")
    return g[group + ["hour", child, "share"]]


# ---- 1. units within positions ----------------------------------------------
unit_info = (o.groupby("unit")
               .agg(position=("position", "first"), floor=("floor", "first"),
                    service_min=("service_min", "mean"))
               .reset_index())
us = smoothed_shares(o, ["position"], "unit")
u = fc.merge(us, on=["position", "hour"], how="left")
u["pred_orders"] = u.pred_orders * u.share
u["pred_patients"] = u.pred_patients * u.share
u = u.merge(unit_info[["unit", "floor", "service_min"]], on="unit", how="left")
u["workload_hours"] = u.pred_orders * u.service_min / 60
u = u.rename(columns={"share": "unit_share"})
u = u[["run_date", "model", "demand_hour", "lead_days", "position", "floor", "unit",
       "unit_share", "pred_orders", "pred_patients", "service_min", "workload_hours"]]
u.to_csv(PROC / "forecast_by_unit_next_week.csv", index=False)

# ---- 2. order types within units --------------------------------------------
ts = smoothed_shares(o, ["unit"], "order_type")
ut = u.assign(hour=u.demand_hour.dt.hour).merge(ts, on=["unit", "hour"], how="left")
ut["pred_orders_type"] = ut.pred_orders * ut.share
ut[["run_date", "demand_hour", "position", "floor", "unit", "order_type", "share",
    "pred_orders_type"]].to_csv(PROC / "forecast_by_unit_ordertype_next_week.csv", index=False)

# ---- 3. floors ---------------------------------------------------------------
fl = (u.groupby(["run_date", "demand_hour", "floor"], as_index=False)
        [["pred_orders", "pred_patients", "workload_hours"]].sum())
fl.to_csv(PROC / "forecast_by_floor_next_week.csv", index=False)

# ---- 4. checks and summary ---------------------------------------------------
print(f"Units: {u.unit.nunique()}  Floors: {u.floor.nunique()}  Unit-hour rows: {len(u)}")
print(f"Check unit total vs position total (should be 0): "
      f"{u.pred_orders.sum() - fc.pred_orders.sum():.6f}")
print(f"Check unit x type total vs unit total (should be 0): "
      f"{ut.pred_orders_type.sum() - u.pred_orders.sum():.6f}")
print(f"Workload with unit-specific service times: {u.workload_hours.sum():.1f} h "
      f"(position-average method: {fc.workload_hours.sum():.1f} h)")

wk = (u.groupby(["position", "unit", "floor"], as_index=False)
        [["pred_orders", "workload_hours"]].sum()
        .sort_values("pred_orders", ascending=False))
print("\nWeekly forecast by unit:"); print(wk.round(1).to_string(index=False))
fw = u.groupby("floor")[["pred_orders", "workload_hours"]].sum()
print("\nWeekly forecast by floor:"); print(fw.round(1).to_string())

# ---- 5. figure ---------------------------------------------------------------
top = wk.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top.unit + "  (" + top.position + ")", top.pred_orders)
ax.set(title="Forecast orders by unit, week of forecast (top 20 units)",
       xlabel="Forecast orders for the week")
fig.tight_layout(); fig.savefig(FIG / "11_forecast_by_unit.png", dpi=120); plt.close(fig)
print("\nDone. Files written to processed/")