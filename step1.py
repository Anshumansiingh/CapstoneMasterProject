"""
IPhSO - Step 1: Data loading, cleaning, and integration
========================================================
Run from the folder that contains "dynamic data/" and "static data/".
Outputs go to ./processed/

What this script does (in order):
  1. Load every file, fixing known format problems.
  2. Clean the orders table (the forecasting target).
  3. Map each order to a physical location (floor / wing / position).
  4. Build the hourly demand table  -> input to the forecasting model.
  5. Clean the MLA schedule          -> input to the optimization model.
  6. Write a data-quality report so every decision is documented.

Items marked  # CONFIRM  are assumptions to check with your professor.
"""
from pathlib import Path
import re
import pandas as pd
import numpy as np

DYN, STA, OUT = Path("dynamic data"), Path("static data"), Path("processed")
OUT.mkdir(exist_ok=True)
report = []  # data-quality log, written at the end


def log(msg):
    print(msg)
    report.append(msg)


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
orders = pd.read_csv(DYN / "phlebotomy_orders.csv")
mla_dir = pd.read_csv(DYN / "mla-directory.csv")

# The schedule has NO header row. Column meanings inferred from values.  # CONFIRM
SCHED_COLS = [
    "person_id", "employee_id", "tech_code",   # hashed IDs
    "employment_status",                       # F / P / C  (full, part, casual)
    "fte",                                     # 1.0, 0.7, ...
    "department", "position_desc",
    "shift_date", "start_time", "end_time",
    "availability_reason",                     # Available / SICK / Leave ...
    "shift_code",                              # e.g. 700D8P
    "time_code",                               # WRK, RSK, USK, STAT ...
    "pay_type",                                # REG / UNPAID / OT1.0
    "position_override_flag",                  # Y / N
]
sched = pd.read_csv(DYN / "phc_schedule.csv")
sched.columns = SCHED_COLS

positions = pd.read_csv(STA / "positions.csv").drop_duplicates()
svc = pd.read_csv(STA / "location-service-times.csv")
# Header says floor,wing,max,min but every row has max < min -> labels swapped.
svc = svc.rename(columns={"max": "min_minutes", "min": "max_minutes"})  # CONFIRM
svc["mean_minutes"] = (svc.min_minutes + svc.max_minutes) / 2

params = pd.read_csv(STA / "parameters.csv", header=None, index_col=0)[1].to_dict()

# sph-locations.csv has broken line endings (\r\r) that merge two rows; repair.
raw = (STA / "sph-locations.csv").read_bytes().decode("utf-8", "ignore")
lines = [l.strip() for l in re.split(r"[\r\n]+", raw) if l.strip()]
sph_loc = pd.DataFrame([l.rsplit(",", 1) for l in lines[1:]],
                       columns=["ORDER_LOCATION", "LOCATION_TYPE"])

log(f"Loaded orders={orders.shape}, schedule={sched.shape}, "
    f"mla_dir={mla_dir.shape}, positions={positions.shape}")

# ---------------------------------------------------------------------------
# 2. CLEAN ORDERS
# ---------------------------------------------------------------------------
for c in ["order_date", "scheduled_date", "collection_date", "received_date"]:
    orders[c] = pd.to_datetime(orders[c])
orders = orders.rename(columns={"collection_to_received(min)": "collect_to_recv_min"})

n0 = len(orders)
# 2a. Keep only St Paul's Hospital collections (other labs are a handful of rows)
orders = orders[orders.collection_lab == "St Paul's Hospital"]
# 2b. Drop outpatient lab (SPHLAB): no room, daytime-only -> out of scope (Sec 1.7)
orders["is_outpatient_lab"] = orders.patient_ordering_location.eq("SPHLAB")
log(f"SPHLAB outpatient orders removed: {orders.is_outpatient_lab.sum()}")
orders = orders[~orders.is_outpatient_lab]
# 2c. Drop non-SPH ordering locations (other hospitals / clinics)
orders = orders[orders.patient_ordering_location.str.startswith("SPH")]
log(f"Orders kept after scope filters: {len(orders)} of {n0}")

# 2d. Last day in the extract is partial -> drop it so it is not read as low demand
last_day = orders.collection_date.dt.normalize().max()
if (orders.collection_date.dt.normalize() == last_day).sum() < 200:
    orders = orders[orders.collection_date.dt.normalize() < last_day]
    log(f"Dropped partial final day {last_day.date()}")

# 2e. Flag (don't silently delete) suspicious timings
orders["lead_min"] = (orders.collection_date - orders.order_date).dt.total_seconds() / 60
orders["flag_collect_before_order"] = orders.lead_min < 0
orders["flag_long_transport"] = orders.collect_to_recv_min > 240
log(f"Collected before ordered: {orders.flag_collect_before_order.sum()}; "
    f"collection->received > 4h: {orders.flag_long_transport.sum()}")

# 2f. Priority: multi-valued strings like "R, U" or "A, T,ATTN"
#     Split into tokens; keep the highest-urgency one as the order type.
PRIORITY_RANK = ["S", "U", "T", "A", "R"]   # CONFIRM meanings + ranking
def primary_priority(p):
    toks = set(re.findall(r"[A-Z]+", str(p)))
    for k in PRIORITY_RANK:
        if k in toks:
            return k
    return "OTHER"
orders["priority_primary"] = orders.priority.map(primary_priority)
orders["order_type"] = orders.phlebotomy_code + "_" + orders.priority_primary
log("Priority distribution: " + orders.priority_primary.value_counts().to_dict().__str__())

# 2g. Was the draw done by an MLA in the directory (vs e.g. a nurse)?
orders["by_mla"] = orders.technologist_code.isin(mla_dir.tech_code)
log(f"Share of orders collected by a directory MLA: {orders.by_mla.mean():.1%}")

# ---------------------------------------------------------------------------
# 3. MAP ORDERS TO PHYSICAL LOCATION
# ---------------------------------------------------------------------------
# The ORDERING location is not always where the patient is (ED-hold patients
# sit on 7A-7D, etc.). The MLA must walk to the ROOM, so parse the room first
# and fall back to the ordering location.
ROOM_PREFIX = {          # room prefix -> (floor, wing)   # CONFIRM with layout
    "PED": (1, "ED"), "PEH": (1, "EH"), "PICU": (3, "ICU"), "PCIC": (3, "CICU"),
    "PCSI": (3, "CSICU"), "PNIC": (3, "NICU"), "PPAR": (3, "PAR"),
    "PSDC": (3, "SDC"), "P3M": (3, "MC"), "P2N": (2, "N"), "P2E": (2, "E"),
    "P4N": (4, "NW"), "P5CS": (5, "CD"), "PSU5": (5, "SUSU"), "PSHA": (1, "SHAU"),
}
ORDER_LOC = {            # ordering location -> (floor, wing) fallback
    "SPHED": (1, "ED"), "SPHEH": (1, "EH"), "SPHICU": (3, "ICU"),
    "SPHCICU": (3, "CICU"), "SPHCSICU": (3, "CSICU"), "SPHNICU": (3, "NICU"),
    "SPHPAR": (3, "PAR"), "SPHPAC": (3, "PAC"), "SPHSDC": (3, "SDC"),
    "SPH3MC": (3, "MC"), "SPH2N": (2, "N"), "SPHPASU": (2, "PASU"),
    "SPH4NW": (4, "NW"), "SPH5CD": (5, "CD"), "SPHSUSU": (5, "SUSU"),
    "SPHSHAU": (1, "SHAU"),
}

def locate(row):
    room, loc = str(row.room), row.patient_ordering_location
    m = re.match(r"^P(\d{1,2})([A-D])\d", room)           # e.g. P7D04002
    if m:
        return int(m.group(1)), m.group(2), "room"
    for pre, fw in sorted(ROOM_PREFIX.items(), key=lambda x: -len(x[0])):
        if room.startswith(pre):
            return fw + ("room",)
    m = re.match(r"^SPH(\d{1,2})([A-D]{1,2})$", loc)        # e.g. SPH7D
    if m:
        return int(m.group(1)), m.group(2), "order_loc"
    if loc in ORDER_LOC:
        return ORDER_LOC[loc] + ("order_loc",)
    return np.nan, None, "unmapped"

orders[["floor", "wing", "loc_source"]] = orders.apply(locate, axis=1, result_type="expand")
orders = orders.merge(positions, on=["floor", "wing"], how="left")
log("Location source: " + orders.loc_source.value_counts().to_dict().__str__())
log(f"Orders with no MLA position after mapping: {orders.position.isna().sum()} "
    f"(top codes: {orders[orders.position.isna()].room.str[:4].value_counts().head(8).to_dict()})")

# Service time (minutes per draw) for workload estimation
orders = orders.merge(svc[["floor", "wing", "mean_minutes"]], on=["floor", "wing"], how="left")
default_min = svc.mean_minutes.median()
orders["service_min"] = orders.mean_minutes.fillna(default_min)
log(f"Orders using default service time ({default_min} min): {orders.mean_minutes.isna().sum()}")

orders.to_csv(OUT / "orders_clean.csv", index=False)

# ---------------------------------------------------------------------------
# 4. HOURLY DEMAND TABLE  (forecasting target)
# ---------------------------------------------------------------------------
# Demand time = scheduled_date (when the draw is DUE), not collection_date,
# because collection time is already distorted by how many MLAs were on shift.
orders["demand_hour"] = orders.scheduled_date.dt.floor("h")
# There is no patient ID: unique room per hour is used as a patient proxy.  # CONFIRM
grp = ["demand_hour", "floor", "wing", "position", "order_type"]
demand = (orders.groupby(grp, dropna=False)
          .agg(n_orders=("accession", "size"),
               n_patients=("room", "nunique"),
               workload_hours=("service_min", lambda x: x.sum() / 60))
          .reset_index())

# Fill the zeros: an hour with no orders must appear as 0, not be missing.
keys = demand[["floor", "wing", "position", "order_type"]].drop_duplicates()
hours = pd.DataFrame({"demand_hour": pd.date_range(
    orders.demand_hour.min().normalize(), orders.demand_hour.max().floor("D")
    + pd.Timedelta(hours=23), freq="h")})
full = hours.merge(keys, how="cross").merge(
    demand, on=grp, how="left").fillna({"n_orders": 0, "n_patients": 0, "workload_hours": 0})

# Calendar features
full["hour"] = full.demand_hour.dt.hour
full["dow"] = full.demand_hour.dt.dayofweek
full["is_weekend"] = full.dow >= 5
holidays = pd.to_datetime(["2024-05-20", "2024-07-01"])   # Victoria Day, Canada Day
full["is_holiday"] = full.demand_hour.dt.normalize().isin(holidays)
full.to_csv(OUT / "hourly_demand.csv", index=False)
log(f"Hourly demand table: {full.shape}, share of zero cells {(full.n_orders == 0).mean():.1%}")

# ---------------------------------------------------------------------------
# 5. CLEAN MLA SCHEDULE  (optimization input)
# ---------------------------------------------------------------------------
sched["shift_date"] = pd.to_datetime(sched.shift_date)
sched["start_time"] = pd.to_datetime(sched.start_time)
sched["end_time"] = pd.to_datetime(sched.end_time)
sched["hours"] = (sched.end_time - sched.start_time).dt.total_seconds() / 3600
sched["shift_code"] = sched.shift_code.str.upper()
bad_codes = sched.shift_code.str.contains(r"E\+", na=False)
log(f"Shift codes corrupted by Excel scientific notation (e.g. 1.30E+11): {bad_codes.sum()}")
sched["working"] = sched.availability_reason.eq("Available") & ~sched.time_code.isin(["STAT"])
sched.to_csv(OUT / "schedule_clean.csv", index=False)
log(f"Schedule range {sched.shift_date.min().date()} to {sched.shift_date.max().date()}; "
    f"working rows {sched.working.sum()} of {len(sched)}")

# ID linkage check: does the schedule join to the directory / orders?
for c in ["person_id", "employee_id", "tech_code"]:
    hits = {mc: len(set(sched[c]) & set(mla_dir[mc].dropna())) for mc in mla_dir.columns}
    log(f"schedule.{c} overlaps with directory columns: {hits}")

(OUT / "data_quality_report.txt").write_text("\n".join(report))
print("\nDone. Files written to ./processed/")