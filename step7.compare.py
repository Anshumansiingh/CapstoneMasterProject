"""
IPhSO - Step 7: Comparing optimization approaches (heuristics vs MILP)
=====================================================================
Run AFTER steps 5 and 6, from the same folder.

Compares five ways of deciding where each MLA works:
  1. Rule-based      : each MLA at one position for the whole shift   (from Step 6)
  2. Greedy          : hour-by-hour greedy heuristic with floating     (constructive heuristic)
  3. Hill climbing   : local search starting from the greedy schedule  (local-search heuristic)
  4. Sim. annealing  : simulated annealing from the greedy schedule    (metaheuristic)
  5. MILP (HiGHS)    : the exact model from Step 6 (its schedule is read from opt_assignments.csv)

Fair comparison: every heuristic only decides WHICH positions each MLA covers.
Each schedule is then scored with exactly the same Step 6 model (same rules,
same points system), by fixing the position decisions and letting the model
allocate the hours of work, as Step 6 does for the rule-based schedule.

Reads  : the Step 5 inputs (via the Step 6 script) and processed/opt_assignments.csv
         (the MILP schedule from Step 6, re-scored here exactly like the other methods)
Writes : processed/c2_methods_by_day.csv, processed/c2_methods_summary.csv
         processed/figures/17_c2_methods_comparison.png
         processed/figures/18_c2_annealing_convergence.png
"""
RPT
from pathlib import Path
import math, random, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0. Reuse the Step 6 model (inputs, parameters, solve_day, rule_based, kpis)
# ---------------------------------------------------------------------------
for name in ["step6.optimize.py", "iphso_step6_optimize.py"]:
    if Path(name).exists():
        STEP6 = name; break
else:
    raise FileNotFoundError("Put this script in the same folder as the Step 6 script")
code = open(STEP6, encoding="utf-8").read()
exec(code[:code.index("# 2. SOLVE EACH DAY")], globals())
print(f"Using the model and settings from {STEP6}")

SEED = 42
HC_STALL = 1500        # hill climbing stops after this many tries without improvement
SA_ITERS = 6000        # simulated annealing iterations per day
GREEDY_MAX_POS = 3     # greedy keeps at most this many positions per MLA
PIDX = {p: i for i, p in enumerate(POSITIONS)}


# ---------------------------------------------------------------------------
# 1. Fast scoring used INSIDE the heuristics (a quick approximation of the
#    Step 6 points system; final scores always come from the exact model)
# ---------------------------------------------------------------------------
class Day:
    def __init__(self, day):
        g = sh[sh.day == day]
        self.day, self.S = day, sorted(g.shift_id.unique())
        self.hours = [day + pd.Timedelta(hours=k) for k in range(24)]
        nS, nP = len(self.S), len(POSITIONS)
        self.cap = np.zeros((nS, 24)); self.ben = np.zeros((nS, 24))
        self.allowed = [[set() for _ in range(24)] for _ in range(nS)]
        for r in g.itertuples():
            i, h = self.S.index(r.shift_id), r.hour.hour
            self.cap[i, h] = CAPACITY * r.frac; self.ben[i, h] = BENCHMARK * r.frac
            self.allowed[i][h] = {PIDX[p] for p in r.allowed_positions.split(";")}
        self.allP = [set().union(*a) for a in self.allowed]
        self.avail = self.cap.sum(1) / CAPACITY
        self.main = self.avail >= BALANCE_MIN_HOURS
        self.D = np.array([[DEMAND.get((p, h), 0.0) for h in self.hours] for p in POSITIONS])
        self.need = [(p, h) for p in range(nP) for h in range(24) if self.D[p, h] > 1e-6]
        self.who = {(p, h): [i for i in range(nS) if p in self.allowed[i][h]] for (p, h) in self.need}
        self.hrs = [[h for h in range(24) if self.cap[i, h] > 0] for i in range(nS)]
        self.floor = np.array([FLOOR[p] for p in POSITIONS])

    def score(self, y):
        """Approximate points for an assignment y (list of position sets)."""
        rem = self.D.copy(); work = np.zeros(len(self.S)); over = 0.0
        for h in range(24):
            act = [(len(y[i] & self.allowed[i][h]), i) for i in range(len(self.S)) if self.cap[i, h] > 0]
            for _, i in sorted(act):
                ps = [p for p in y[i] if p in self.allowed[i][h]]
                c = self.cap[i, h]; done = 0.0
                for p in sorted(ps, key=lambda q: -rem[q, h]):
                    t = min(c, rem[p, h]); rem[p, h] -= t; c -= t; done += t
                    if c <= 1e-9: break
                work[i] += done; over += max(0.0, done - self.ben[i, h])
        unmet = rem.sum()
        nocov = sum(1 for k in self.need if not any(k[0] in y[i] for i in self.who[k]))
        nfl = sum(len(s) for s in y)
        travel = sum(FLOOR_CHANGE * (self.floor[list(s)].max() - self.floor[list(s)].min()) for s in y if s)
        m = self.main
        umax = (work[m] / self.avail[m]).max() if m.any() else 0.0
        fair = 0.0
        if FAIRNESS_COST > 0 and m.any():
            W, A = work[m].sum(), self.avail[m].sum()
            fair = np.abs(work[m] - self.avail[m] / A * W).sum()
        return (UNMET_COST * unmet + NO_COVER_COST * nocov + FLOAT_COST * nfl + TRAVEL_COST * travel
                + OVERLOAD_COST * over + BALANCE_COST * umax + FAIRNESS_COST * fair)

    def neighbour(self, y, rng):
        """Small change: add, remove, or swap one position for one MLA."""
        z = [set(s) for s in y]
        i = rng.randrange(len(z)); options = list(self.allP[i])
        move = rng.random()
        if move < 0.35 and len(z[i]) < min(MAX_FLOAT, len(options)):
            z[i].add(rng.choice([p for p in options if p not in z[i]]))
        elif move < 0.6 and len(z[i]) > 1:
            z[i].remove(rng.choice(list(z[i])))
        else:
            free = [p for p in options if p not in z[i]]
            if free and z[i]:
                z[i].remove(rng.choice(list(z[i]))); z[i].add(rng.choice(free))
        return z


# ---------------------------------------------------------------------------
# 2. The three heuristic methods
# ---------------------------------------------------------------------------
def greedy(d):
    """Hour by hour, send each MLA to the allowed position with the most work left."""
    rem = d.D.copy(); y = []
    order = sorted(range(len(d.S)), key=lambda i: d.hrs[i][0] if d.hrs[i] else 99)
    contrib = {i: {} for i in range(len(d.S))}
    for i in order:
        for h in d.hrs[i]:
            if not d.allowed[i][h]: continue
            p = max(d.allowed[i][h], key=lambda q: rem[q, h])
            t = min(d.cap[i, h], rem[p, h]); rem[p, h] -= t
            contrib[i][p] = contrib[i].get(p, 0) + t
    for i in range(len(d.S)):
        top = sorted(contrib[i], key=lambda p: -contrib[i][p])[:GREEDY_MAX_POS]
        top = [p for p in top if contrib[i][p] > 1e-6]
        if not top:   # must assign: busiest allowed position over the shift
            top = [max(d.allP[i], key=lambda p: d.D[p, d.hrs[i]].sum())]
        y.append(set(top))
    return y


def hill_climb(d, y0, rng):
    y, best = y0, d.score(y0); stall = 0
    while stall < HC_STALL:
        z = d.neighbour(y, rng); c = d.score(z)
        if c < best - 1e-9: y, best, stall = z, c, 0
        else: stall += 1
    return y


def annealing(d, y0, rng, trace=None):
    y, cur = y0, d.score(y0); best_y, best = y, cur
    ups = [d.score(d.neighbour(y, rng)) - cur for _ in range(60)]
    ups = [u for u in ups if u > 0] or [1.0]
    T0 = -np.mean(ups) / math.log(0.5)       # accept an average uphill move with p = 0.5
    T_end = T0 * 1e-3
    for k in range(SA_ITERS):
        T = T0 * (T_end / T0) ** (k / SA_ITERS)
        z = d.neighbour(y, rng); c = d.score(z)
        if c <= cur or rng.random() < math.exp(-(c - cur) / T):
            y, cur = z, c
            if cur < best: best_y, best = y, cur
        if trace is not None and k % 50 == 0: trace.append((k, cur, best))
    return best_y


# ---------------------------------------------------------------------------
# 3. Run every method, then score each schedule with the exact Step 6 model
# ---------------------------------------------------------------------------
rng = random.Random(SEED)
rows, trace_day = [], []
mil = pd.read_csv(PROC / "opt_assignments.csv")
mil["day"] = pd.to_datetime(mil.day)
for day in DAYS:
    d = Day(day)
    to_fixed = lambda y: {d.S[i]: {POSITIONS[p] for p in y[i]} for i in range(len(d.S))}
    t0 = time.time(); fx = rule_based(day); t_rb = time.time() - t0
    t0 = time.time(); yg = greedy(d); t_g = time.time() - t0
    t0 = time.time(); yh = hill_climb(d, yg, rng); t_h = time.time() - t0 + t_g
    tr = [] if day == DAYS[0] else None
    t0 = time.time(); ya = annealing(d, yg, rng, tr); t_a = time.time() - t0 + t_g
    if tr is not None: trace_day = tr
    md = mil[mil.day == day]
    fm = {r.shift_id: set(str(r.positions).split(" + ")) for r in md.itertuples()}
    for label, fixed, secs in [("Rule-based", fx, t_rb), ("Greedy", to_fixed(yg), t_g),
                               ("Hill climbing", to_fixed(yh), t_h), ("Simulated annealing", to_fixed(ya), t_a),
                               ("MILP (HiGHS)", fm, np.nan)]:
        r = solve_day(day, fixed=fixed)                 # exact scoring, same model as Step 6
        k = kpis(day, r, label); k["points"] = round(r["objective"], 1); k["seconds"] = round(secs, 2)
        rows.append(k)
    print(f"{pd.Timestamp(day).date()}  " + "  ".join(f"{r['schedule'][:6]} {r['coverage_%']}%" for r in rows[-5:]))

res = pd.DataFrame(rows)
res.to_csv(PROC / "c2_methods_by_day.csv", index=False)

order = ["Rule-based", "Greedy", "Hill climbing", "Simulated annealing", "MILP (HiGHS)"]
wk = res.groupby("schedule").agg(unmet_hours=("unmet_hours", "sum"), demand_hours=("demand_hours", "sum"),
                                 unmet_7to10_hours=("unmet_7to10_hours", "sum"),
                                 uncovered_position_hours=("uncovered_position_hours", "sum"),
                                 floating_mla_shifts=("floating_mlas", "sum"), travel_hours=("travel_hours", "sum"),
                                 max_util_main=("max_util_main", "max"), sd_util_main=("sd_util_main", "mean"),
                                 points=("points", "sum"), seconds_per_week=("seconds", "sum")).reindex(order)
wk["coverage_%"] = (100 * (1 - wk.unmet_hours / wk.demand_hours)).round(1)
wk.loc["MILP (HiGHS)", "seconds_per_week"] = np.nan     # measured in Step 6 (about 30 minutes for the week)
best = wk.points.min(); wk["points_vs_best_%"] = (100 * (wk.points - best) / best).round(1)
wk.round(2).to_csv(PROC / "c2_methods_summary.csv")
print("\n=== Week summary by method ===")
print(wk[["coverage_%", "unmet_hours", "unmet_7to10_hours", "uncovered_position_hours", "floating_mla_shifts",
          "travel_hours", "max_util_main", "sd_util_main", "points", "points_vs_best_%", "seconds_per_week"]].round(2).to_string())

# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
FIG.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
cols = ["#E08A2B", "#9AA5AA", "#7FB7BE", "#2A8C99", "#0B4F5C"]
b = ax[0].bar(order, wk["coverage_%"], color=cols); ax[0].bar_label(b, fmt="%.1f%%", fontsize=8)
ax[0].set(ylim=(0, 105), ylabel="Workload covered (%)", title="Coverage by method")
b = ax[1].bar(order, wk["unmet_hours"], color=cols); ax[1].bar_label(b, fmt="%.1f", fontsize=8)
ax[1].set(ylabel="Unmet workload (hours)", title="Work left undone by method")
for a in ax: a.tick_params(axis="x", rotation=25, labelsize=8)
fig.tight_layout(); fig.savefig(FIG / "17_c2_methods_comparison.png", dpi=120); plt.close(fig)

if trace_day:
    k, cur, best = zip(*trace_day)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(k, cur, lw=0.8, alpha=0.6, label="Current schedule"); ax.plot(k, best, lw=2, label="Best found")
    ax.set_yscale("log")
    ax.set(xlabel="Iteration", ylabel="Points (approximate, log scale)", title=f"Simulated annealing search, {pd.Timestamp(DAYS[0]).date()}")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "18_c2_annealing_convergence.png", dpi=120); plt.close(fig)
print("\nDone. Results in processed/c2_methods_*.csv and figures 17-18")