"""
IPhSO - Step 6: MLA assignment optimization (Component 2, version 1)
====================================================================
Run AFTER step 5, from the same folder.

Reads  : processed/c2_shifts.csv, c2_shift_hours.csv, c2_demand.csv, c2_positions.csv
         static data/parameters.csv
Writes : processed/opt_assignments.csv   one row per MLA shift-day: positions, workload, utilization
         processed/opt_hourly_plan.csv   shift x position x hour: hours of work assigned
         processed/opt_coverage.csv      position x hour: demand, work done, unmet (both schedules)
         processed/c2_comparison.csv     KPIs per day: optimized vs rule-based schedule
         processed/figures/12_c2_hourly_coverage.png
         processed/figures/13_c2_utilization.png
         processed/figures/14_c2_mlas_by_position.png

Method : mixed-integer linear programming (MILP), solved with HiGHS through
         scipy.optimize.milp (free and open-source; included with Anaconda, so
         no extra installation is needed). One model is solved per day.

Two schedules are evaluated with exactly the same model and measures:
  * OPTIMIZED  : the model chooses where each MLA works (floating allowed).
  * RULE-BASED : each MLA stays at one position for the whole shift, chosen
                 by a simple rule (the busiest position still uncovered). This
                 stands in for a static route until historical assignments
                 are available (partner question 1).
"""
from pathlib import Path
from collections import defaultdict
import time
import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC, STA, FIG = Path("processed"), Path("static data"), Path("processed") / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 0. PARAMETERS (parameters.csv) and model weights
# ---------------------------------------------------------------------------
par = pd.read_csv(STA / "parameters.csv", header=None, index_col=0)[1]
CAPACITY = float(par["mla_hourly_workload_capacity"])      # 1.1 hard max per MLA-hour   # ASSUMPTION
BENCHMARK = float(par["workload_benchmark"])               # 1.0 target per MLA-hour
FLOOR_CHANGE = float(par["floor_change_travel_time"])      # 0.6 per floor apart          # ASSUMPTION
TRAVEL_COST = float(par["travel_workload_cost_per_hour"])  # 10
FLOAT_COST = float(par["float_count_cost"])                # 3 per location covered
UNMET_COST = float(par["constraint_violation_cost"])       # 50 per unmet workload hour
MAX_FLOAT = int(par["max_float_locations"])                # 11
MUST_ASSIGN = str(par["must_assign_mla"]).strip() == "True"
GAP = float(par["optimality_gap"])                         # 0.01
NODES = int(par["max_nodes"])                              # 2000
OVERLOAD_COST = TRAVEL_COST    # per hour of work above the 1.0 benchmark              # ASSUMPTION
BALANCE_COST = 20.0            # on the busiest main collector's utilization           # ASSUMPTION
BALANCE_MIN_HOURS = 4          # shifts collecting >= 4 h count for balance            # ASSUMPTION
FAIRNESS_COST = 5.0            # per workload-hour an MLA is above or below their fair  # ASSUMPTION
                               # share (set to 0 to switch the fairness term off)
NO_COVER_COST = 1000.0         # position-hour with demand but nobody assigned (hard rule)
TIME_LIMIT = 120               # seconds per day

# ---------------------------------------------------------------------------
# 1. INPUTS FROM STEP 5
# ---------------------------------------------------------------------------
shifts = pd.read_csv(PROC / "c2_shifts.csv", parse_dates=["start_time", "end_time"]).set_index("shift_id")
sh = pd.read_csv(PROC / "c2_shift_hours.csv", parse_dates=["hour"])
dem = pd.read_csv(PROC / "c2_demand.csv", parse_dates=["hour"])
pos = pd.read_csv(PROC / "c2_positions.csv")
POSITIONS = list(pos.position)
FLOOR = dict(zip(pos.position, pos.main_floor))
TOPF = float(max(FLOOR.values()))
DEMAND = {(r.position, r.hour): r.workload_smoothed for r in dem.itertuples()}
sh["day"] = sh.hour.dt.normalize()
DAYS = sorted(sh.day.unique())
print(f"{sh.shift_id.nunique()} collector shifts, {len(DAYS)} days, {len(POSITIONS)} positions")


class Model:
    """Small helper: add variables and linear constraints, then solve with HiGHS."""
    def __init__(self):
        self.c, self.lb, self.ub, self.integ = [], [], [], []
        self.r, self.k, self.v, self.clb, self.cub = [], [], [], [], []

    def var(self, cost=0.0, lb=0.0, ub=np.inf, binary=False):
        self.c.append(cost); self.lb.append(lb); self.ub.append(1.0 if binary else ub)
        self.integ.append(1 if binary else 0)
        return len(self.c) - 1

    def con(self, coeffs, lb=-np.inf, ub=np.inf):
        row = len(self.clb)
        for j, val in coeffs.items():
            self.r.append(row); self.k.append(j); self.v.append(val)
        self.clb.append(lb); self.cub.append(ub)

    def solve(self):
        A = coo_matrix((self.v, (self.r, self.k)), shape=(len(self.clb), len(self.c))).tocsr()
        return milp(c=np.array(self.c), integrality=np.array(self.integ), bounds=Bounds(self.lb, self.ub),
                    constraints=LinearConstraint(A, self.clb, self.cub),
                    options={"mip_rel_gap": GAP, "time_limit": TIME_LIMIT, "node_limit": NODES, "disp": False})


def solve_day(day, fixed=None):
    """Build and solve one day. fixed = {shift_id: set(positions)} for the rule-based schedule."""
    d_sh = sh[sh.day == day]
    hours = [day + pd.Timedelta(hours=k) for k in range(24)]
    m = Model()
    y, x, z = {}, defaultdict(dict), {}
    avail = {}
    for sid, g in d_sh.groupby("shift_id"):
        frac = dict(zip(g.hour, g.frac))
        allowed = {h: set(a.split(";")) for h, a in zip(g.hour, g.allowed_positions)}
        avail[sid] = sum(frac.values())
        for p in POSITIONS:
            ok = [h for h in frac if p in allowed[h]]
            if not ok:
                continue
            fix = None if fixed is None else (1.0 if p in fixed.get(sid, set()) else 0.0)
            y[sid, p] = m.var(cost=FLOAT_COST, binary=True) if fix is None else m.var(cost=FLOAT_COST, lb=fix, ub=fix)
            for h in ok:                                          # work only where assigned
                x[sid][p, h] = m.var(ub=CAPACITY * frac[h])
                m.con({x[sid][p, h]: 1, y[sid, p]: -CAPACITY * frac[h]}, ub=0)
        for h, f in frac.items():                                 # hourly capacity and overload
            xs = [v for (p, hh), v in x[sid].items() if hh == h]
            if xs:
                m.con({v: 1 for v in xs}, ub=CAPACITY * f)
                ov = m.var(cost=OVERLOAD_COST)
                m.con({**{v: 1 for v in xs}, ov: -1}, ub=BENCHMARK * f)
        ys = [y[sid, p] for p in POSITIONS if (sid, p) in y]
        if ys and fixed is None:
            m.con({v: 1 for v in ys}, lb=1 if MUST_ASSIGN else 0, ub=MAX_FLOAT)
        # travel: the MLA walks the span between the highest and lowest floor covered
        # travel hours = floor_change_travel_time x (top floor - bottom floor)        # ASSUMPTION
        if ys:
            top, bottom = m.var(lb=0, ub=TOPF), m.var(lb=0, ub=TOPF)
            z[sid] = (top, bottom)
            m.c[top] += TRAVEL_COST * FLOOR_CHANGE
            m.c[bottom] -= TRAVEL_COST * FLOOR_CHANGE
            for p in POSITIONS:
                if (sid, p) in y:
                    m.con({top: 1, y[sid, p]: -FLOOR[p]}, lb=0)                  # top >= floor of p if assigned
                    m.con({bottom: 1, y[sid, p]: TOPF}, ub=FLOOR[p] + TOPF)       # bottom <= floor of p if assigned
    unmet, nocov = {}, {}
    for p in POSITIONS:
        for h in hours:
            dmd = DEMAND.get((p, h), 0.0)
            xs = [x[s][p, h] for s in x if (p, h) in x[s]]
            unmet[p, h] = m.var(cost=UNMET_COST)                  # work done + unmet = demand
            m.con({**{v: 1 for v in xs}, unmet[p, h]: 1}, lb=dmd, ub=dmd)
            if dmd > 1e-6:                                        # someone assigned and on duty
                ys = [y[s, p] for s in x if (p, h) in x[s]]
                nocov[p, h] = m.var(cost=NO_COVER_COST, ub=1)
                m.con({**{v: 1 for v in ys}, nocov[p, h]: 1}, lb=1)
    umax = m.var(cost=BALANCE_COST)                               # workload balance
    main = [sid for sid, xv in x.items() if xv and avail[sid] >= BALANCE_MIN_HOURS]
    for sid in main:
        m.con({**{v: 1 / avail[sid] for v in x[sid].values()}, umax: -1}, ub=0)
    # fairness: each main collector's fair share of the day's work is proportional
    # to their collection hours:  fair_s = (avail_s / total avail) x total work
    # dev_s >= |work_s - fair_s|, and every hour of deviation costs FAIRNESS_COST
    if FAIRNESS_COST > 0 and main:
        total_avail = sum(avail[s] for s in main)
        total = m.var()                                           # total work of main collectors
        m.con({**{v: 1 for s in main for v in x[s].values()}, total: -1}, lb=0, ub=0)
        for sid in main:
            share = avail[sid] / total_avail
            dev = m.var(cost=FAIRNESS_COST)
            work = {v: 1 for v in x[sid].values()}
            m.con({**{v: -1 for v in work}, total: share, dev: 1}, lb=0)   # dev >= work - fair
            m.con({**work, total: -share, dev: 1}, lb=0)                    # dev >= fair - work
    res = m.solve()
    if res.x is None:
        raise RuntimeError(f"{pd.Timestamp(day).date()}: {res.message}")
    sol = res.x
    out = {"y": {k: sol[v] > 0.5 for k, v in y.items()}, "x": {(s, p, h): sol[v] for s in x for (p, h), v in x[s].items()},
           "unmet": {k: sol[v] for k, v in unmet.items()}, "nocov": {k: sol[v] > 0.5 for k, v in nocov.items()},
           # travel measured from the floors actually covered (the helper variables T and B
           # can be left loose when the solver stops at a time or node limit)
           "travel": sum(FLOOR_CHANGE * (max(FLOOR[p] for p in ps) - min(FLOOR[p] for p in ps))
                         for ps in [[p for (s2, p), v in y.items() if s2 == s1 and sol[v] > 0.5] for s1 in {k[0] for k in y}] if ps),
           "avail": avail, "status": res.message.split(".")[0], "objective": res.fun}
    return out


def rule_based(day):
    """One position per MLA for the whole shift: rule shifts keep their listed positions;
    others go to the position with the most demand still uncovered during their shift."""
    d_sh = sh[sh.day == day].sort_values("hour")
    remaining = {k: v for k, v in DEMAND.items() if pd.Timestamp(k[1]).normalize() == day}
    fixed = {}
    for sid, g in d_sh.groupby("shift_id", sort=False):
        allowed_all = set().union(*[set(a.split(";")) for a in g.allowed_positions])
        if shifts.loc[sid, "collector_type"] == "rule" and len(allowed_all) < len(POSITIONS):
            choice = allowed_all if len(allowed_all) <= 2 else {max(allowed_all, key=lambda p: sum(remaining.get((p, h), 0) for h in g.hour))}
        else:
            choice = {max(allowed_all, key=lambda p: sum(remaining.get((p, h), 0) for h in g.hour))}
        for p in choice:
            for h, f in zip(g.hour, g.frac):
                remaining[(p, h)] = max(0.0, remaining.get((p, h), 0) - CAPACITY * f / len(choice))
        fixed[sid] = choice
    return fixed


def kpis(day, r, label):
    d_sh = sh[sh.day == day]
    dmd = sum(v for (p, h), v in DEMAND.items() if pd.Timestamp(h).normalize() == day)
    work = defaultdict(float)
    for (s, p, h), v in r["x"].items():
        work[s] += v
    util = {s: work[s] / r["avail"][s] for s in r["avail"] if r["avail"][s] > 0}
    main = [u for s, u in util.items() if r["avail"][s] >= BALANCE_MIN_HOURS]
    npos = defaultdict(int)
    for (s, p), on in r["y"].items():
        npos[s] += on
    unmet = sum(r["unmet"].values())
    mains = [s for s in util if r["avail"][s] >= BALANCE_MIN_HOURS]
    tw, ta = sum(work[s] for s in mains), sum(r["avail"][s] for s in mains)
    fair_gap = float(np.mean([abs(work[s] - r["avail"][s] / ta * tw) for s in mains])) if mains else 0
    peak = sum(v for (p, h), v in r["unmet"].items() if 7 <= pd.Timestamp(h).hour <= 9)
    return {"day": pd.Timestamp(day).date(), "schedule": label, "mla_shifts": d_sh.shift_id.nunique(),
            "demand_hours": round(dmd, 1), "unmet_hours": round(unmet, 2),
            "coverage_%": round(100 * (1 - unmet / dmd), 1) if dmd else 100.0,
            "unmet_7to10_hours": round(peak, 2), "uncovered_position_hours": int(sum(r["nocov"].values())),
            "floating_mlas": sum(n > 1 for n in npos.values()), "travel_hours": round(r["travel"], 2),
            "max_util_main": round(max(main), 3) if main else 0, "sd_util_main": round(float(np.std(main)), 3) if main else 0,
            "mean_util": round(float(np.mean(list(util.values()))), 3),
            "fair_gap_hours": round(fair_gap, 2), "status": r["status"]}


# ---------------------------------------------------------------------------
# 2. SOLVE EACH DAY: optimized and rule-based
# ---------------------------------------------------------------------------
comp, assign_rows, plan_rows, cov_rows, base_util = [], [], [], [], []
# Optional resume: set environment variable IPHSO_CACHE=1 to save each solved day in
# processed/c2_cache/ and reuse it if the run is interrupted. Delete that folder
# whenever the inputs from Step 5 change.
CACHE = PROC / "c2_cache" if os.environ.get("IPHSO_CACHE") == "1" else None
if CACHE:
    CACHE.mkdir(exist_ok=True)
for day in DAYS:
    t0 = time.time()
    cf = CACHE / f"{pd.Timestamp(day).date()}.pkl" if CACHE else None
    if cf and cf.exists():
        opt, base = pickle.load(open(cf, "rb"))
    else:
        opt = solve_day(day)
        base = solve_day(day, fixed=rule_based(day))
        if cf:
            pickle.dump((opt, base), open(cf, "wb"))
    for r, label in [(opt, "optimized"), (base, "rule-based")]:
        comp.append(kpis(day, r, label))
    bw = defaultdict(float)
    for (s, p, h), v in base["x"].items():
        bw[s] += v
    base_util += [bw[s] / base["avail"][s] for s in base["avail"] if base["avail"][s] >= BALANCE_MIN_HOURS]
    print(f"{pd.Timestamp(day).date()}  optimized coverage {comp[-2]['coverage_%']}% vs rule-based "
          f"{comp[-1]['coverage_%']}%  ({time.time() - t0:.0f} s)")
    work = defaultdict(float)
    for (s, p, h), v in opt["x"].items():
        work[s, p] += v
        if v > 1e-4:
            plan_rows.append({"day": pd.Timestamp(day).date(), "shift_id": s, "position": p, "hour": h, "work_hours": round(v, 3)})
    for s, a in opt["avail"].items():
        ps = [p for p in POSITIONS if opt["y"].get((s, p))]
        info = shifts.loc[s]
        total = sum(work[s, p] for p in ps)
        assign_rows.append({"day": pd.Timestamp(day).date(), "shift_id": s, "person_id": info.person_id,
                            "shift_code": info.shift_code, "role": info.position_desc, "collector_type": info.collector_type,
                            "start": info.start_time, "end": info.end_time, "collect_hours": round(a, 2),
                            "positions": " + ".join(ps), "n_positions": len(ps),
                            "floors": " + ".join(str(FLOOR[p]) for p in ps),
                            "work_by_position": "; ".join(f"{p}: {work[s, p]:.2f} h" for p in ps),
                            "workload_hours": round(total, 2), "utilization": round(total / a, 3) if a else 0})
    for p in POSITIONS:
        for k in range(24):
            h = day + pd.Timedelta(hours=k)
            dmd = DEMAND.get((p, h), 0.0)
            cov_rows.append({"position": p, "hour": h, "demand_hours": round(dmd, 3),
                             "done_optimized": round(dmd - opt["unmet"][p, h], 3),
                             "done_rule_based": round(dmd - base["unmet"][p, h], 3),
                             "mlas_optimized": sum(opt["y"].get((s, p), False) and (s, p, h) in opt["x"] for s in opt["avail"]),
                             "mlas_rule_based": sum(base["y"].get((s, p), False) and (s, p, h) in base["x"] for s in base["avail"])})

comp = pd.DataFrame(comp); cov = pd.DataFrame(cov_rows)
comp.to_csv(PROC / "c2_comparison.csv", index=False)
pd.DataFrame(assign_rows).to_csv(PROC / "opt_assignments.csv", index=False)
pd.DataFrame(plan_rows).to_csv(PROC / "opt_hourly_plan.csv", index=False)
cov.to_csv(PROC / "opt_coverage.csv", index=False)

# ---------------------------------------------------------------------------
# 3. WEEKLY SUMMARY
# ---------------------------------------------------------------------------
wk = comp.groupby("schedule").agg(demand_hours=("demand_hours", "sum"), unmet_hours=("unmet_hours", "sum"),
                                  unmet_7to10_hours=("unmet_7to10_hours", "sum"),
                                  uncovered_position_hours=("uncovered_position_hours", "sum"),
                                  floating_mla_shifts=("floating_mlas", "sum"), travel_hours=("travel_hours", "sum"),
                                  max_util_main=("max_util_main", "max"), sd_util_main=("sd_util_main", "mean"),
                                  fair_gap_hours=("fair_gap_hours", "mean"))
wk["coverage_%"] = (100 * (1 - wk.unmet_hours / wk.demand_hours)).round(1)
print("\n=== Week summary: optimized vs rule-based ===")
print(wk.round(2).T.to_string())

# ---------------------------------------------------------------------------
# 4. FIGURES
# ---------------------------------------------------------------------------
cov["hod"] = cov.hour.dt.hour
hh = cov.groupby("hod")[["demand_hours", "done_optimized", "done_rule_based"]].sum() / len(DAYS)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(hh.index, hh.demand_hours, color="black", lw=2, label="Forecast workload (spread 7–10 a.m.)")
ax.plot(hh.index, hh.done_optimized, marker="o", label="Covered: optimized")
ax.plot(hh.index, hh.done_rule_based, marker="s", ls="--", label="Covered: rule-based")
ax.set(title="Average workload covered by hour of day, all positions", xlabel="Hour", ylabel="Workload hours", xticks=range(24))
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "12_c2_hourly_coverage.png", dpi=120); plt.close(fig)

a = pd.DataFrame(assign_rows)
main = a[a.collect_hours >= BALANCE_MIN_HOURS]
fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot([base_util, list(main.utilization)])
ax.set_xticks([1, 2]); ax.set_xticklabels(["Rule-based", "Optimized"])
ax.set(title="Utilization of main collectors (workload ÷ collection hours)", ylabel="Utilization")
fig.tight_layout(); fig.savefig(FIG / "13_c2_utilization.png", dpi=120); plt.close(fig)

mon = cov[cov.hour.dt.normalize() == DAYS[0]].pivot(index="position", columns="hod", values="mlas_optimized")
fig, ax = plt.subplots(figsize=(12, 4.5))
im = ax.imshow(mon.values, aspect="auto", cmap="Blues", vmin=0)
for (i, j), v in np.ndenumerate(mon.values):
    ax.text(j, i, int(v), ha="center", va="center", fontsize=7, color="white" if v >= 3 else "black")
ax.set(yticks=range(len(mon)), yticklabels=mon.index, xticks=range(24), xlabel="Hour",
       title=f"MLAs assigned per position, {pd.Timestamp(DAYS[0]).date()} (optimized)")
fig.colorbar(im, ax=ax, label="MLAs"); fig.tight_layout(); fig.savefig(FIG / "14_c2_mlas_by_position.png", dpi=120); plt.close(fig)
print("\nDone. Results in processed/ (opt_*.csv, c2_comparison.csv) and figures 12-14")