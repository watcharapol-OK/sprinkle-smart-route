# =====================================================================================
#  SMART ROUTE REBALANCER — MULTI-DEPOT ENTERPRISE EDITION v4.7
#  Fixed Flow: 1. Load Sheet -> 2. Select Depot First -> 3. Column Mapping & Fleet
# =====================================================================================
from __future__ import annotations

import base64
import json
import math
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from folium import plugins

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

st.set_page_config(
    page_title="Smart Route Rebalancer Enterprise",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================================
#  SECTION 1 — CONSTANTS & DOMAIN MODEL
# =====================================================================================
WORKING_DAYS = 6
WEEKS_PER_MONTH = 4.333
DAYS_PER_MONTH = WORKING_DAYS * WEEKS_PER_MONTH

DEFAULT_MONTHLY_CAPACITY = 4160.0
DEFAULT_DAILY_CONTROL_CAP = 156.0

OVERFLOW_LABEL = "ส่วนเกิน (Overflow)"
NO_TRUCK_TOKENS = {"", "nan", "none", "null", "-", "ไม่ระบุ", "na", "n/a"}

OPTIMAL_MIN = 140
OPTIMAL_MAX = 155
AVOID_MIN = 121
AVOID_MAX = 139
TARGET_DAY_CAP = 148

EARTH_RADIUS_M = 6371008.8

DAY_NAMES = {0: "จันทร์", 1: "อังคาร", 2: "พุธ", 3: "พฤหัสบดี", 4: "ศุกร์", 5: "เสาร์"}
DAY_SHORT = {0: "จ", 1: "อ", 2: "พ", 3: "พฤ", 4: "ศ", 5: "ส"}

DAY_TOKENS: List[Tuple[str, int]] = [
    ("จันทร์", 0), ("อังคาร", 1), ("พฤหัสบดี", 3), ("พฤหัสฯ", 3), ("พฤหัส", 3), ("พฤ", 3),
    ("พุธ", 2), ("ศุกร์", 4), ("เสาร์", 5), ("monday", 0), ("tuesday", 1), ("wednesday", 2),
    ("thursday", 3), ("friday", 4), ("saturday", 5), ("mon", 0), ("tue", 1), ("wed", 2),
    ("thu", 3), ("fri", 4), ("sat", 5), ("จ", 0), ("อ", 1), ("พ", 2), ("ศ", 4), ("ส", 5),
]
ALL_DAYS_TOKENS = ("ทุกวัน", "จ-ส", "จันทร์-เสาร์", "จ.-ส.", "ทุกวันทำการ")

# =====================================================================================
#  SECTION 2 — PURE HELPERS
# =====================================================================================
def parse_days_from_string(val_str) -> Tuple[List[int], str]:
    raw = "" if val_str is None else str(val_str)
    val = raw.strip().lower()
    if val in NO_TRUCK_TOKENS:
        return [], "empty"
    compact = re.sub(r"\s+", "", val)
    if any(token in compact for token in ALL_DAYS_TOKENS):
        return list(range(WORKING_DAYS)), "ok"
    days = set()
    tokens = re.split(r"[,|/+\-;\s]+", val)
    for token in tokens:
        token = token.strip().strip(".")
        if not token:
            continue
        if token.isdigit():
            number = int(token)
            if 1 <= number <= WORKING_DAYS:
                days.add(number - 1)
            continue
        for name, day_index in DAY_TOKENS:
            if name in token:
                days.add(day_index)
                break
    return (sorted(days), "ok") if days else ([], "unparsed")

def format_days_to_string(days_list: Sequence[int]) -> str:
    if not days_list:
        return "ไม่ระบุ"
    days = sorted({int(d) for d in days_list if 0 <= int(d) < WORKING_DAYS})
    if not days:
        return "ไม่ระบุ"
    if len(days) == WORKING_DAYS:
        return "จ-ส"
    return ", ".join(DAY_NAMES[d] for d in days)

def format_days_short(days_list: Sequence[int]) -> str:
    if not days_list:
        return "-"
    days = sorted({int(d) for d in days_list if 0 <= int(d) < WORKING_DAYS})
    if not days:
        return "-"
    if len(days) == WORKING_DAYS:
        return "จ-ส"
    return "".join(DAY_SHORT[d] for d in days)

def project_xy(lat, lon, lat0: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    if lat_arr.size == 0 or lon_arr.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), 0.0 if lat0 is None else float(lat0)
    if lat0 is None:
        lat0 = float(np.nanmean(lat_arr))
    scale = math.cos(math.radians(lat0))
    x = np.radians(lon_arr) * EARTH_RADIUS_M * scale
    y = np.radians(lat_arr) * EARTH_RADIUS_M
    return x, y, lat0

def knn_indices(xy: np.ndarray, k: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    k = int(min(max(0, k), max(0, n - 1)))
    if k <= 0:
        return np.zeros((n, 0), dtype=int)
    if HAS_SCIPY:
        tree = cKDTree(xy)
        _, indices = tree.query(xy, k=k + 1, workers=-1)
        if k == 1:
            indices = np.asarray(indices).reshape(n, 2)
        return np.asarray(indices[:, 1:], dtype=int)
    output = np.zeros((n, k), dtype=int)
    chunk_size = 512
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        distances = ((xy[start:end, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
        local_rows = np.arange(end - start)
        global_rows = np.arange(start, end)
        distances[local_rows, global_rows] = np.inf
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        nearest_distances = np.take_along_axis(distances, nearest, axis=1)
        order = np.argsort(nearest_distances, axis=1)
        output[start:end] = np.take_along_axis(nearest, order, axis=1)
    return output

def kmeans_seeds(xy: np.ndarray, weights: np.ndarray, k: int, iters: int = 20, seed: int = 42) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(xy)
    if n == 0:
        return np.zeros((0, 2), dtype=float)
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    valid_weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1e-9)
    first_index = rng.choice(n, p=valid_weights / valid_weights.sum())
    centers = [xy[first_index]]
    for _ in range(k - 1):
        cur_centers = np.asarray(centers, dtype=float)
        dist_sq = np.min(((xy[:, None, :] - cur_centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        probs = dist_sq * valid_weights
        probs = (probs / probs.sum()) if probs.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(xy[rng.choice(n, p=probs)])
    centers_arr = np.asarray(centers, dtype=float)
    for _ in range(max(1, int(iters))):
        dist_sq = ((xy[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2)
        labels = dist_sq.argmin(axis=1)
        new_centers = centers_arr.copy()
        for c_idx in range(k):
            mask = labels == c_idx
            if mask.any():
                new_centers[c_idx] = np.average(xy[mask], axis=0, weights=valid_weights[mask])
        if np.allclose(new_centers, centers_arr, atol=1e-3, rtol=0.0):
            break
        centers_arr = new_centers
    return centers_arr

def clean_truck_ids(values: Sequence) -> List[str]:
    output = []
    for value in values:
        t_id = str(value).strip()
        if t_id.lower() in NO_TRUCK_TOKENS:
            continue
        output.append(t_id)
    return sorted(set(output))

def guess_col(substrings: Sequence[str], cols: Sequence[str], fallback: Optional[str] = None, allow_none: bool = False) -> Optional[str]:
    for column in cols:
        column_text = str(column).lower()
        if any(str(sub).lower() in column_text for sub in substrings):
            return column
    if allow_none:
        return None
    return fallback if fallback is not None else (cols[0] if len(cols) else None)

# =====================================================================================
#  SECTION 3 — CONFIG / RESULT DATACLASSES
# =====================================================================================
@dataclass
class ZoningConfig:
    lat_col: str
    lon_col: str
    vol_col: str
    truck_col: str
    id_col: str
    day_col: str
    name_col: Optional[str] = None
    monthly_capacity: float = DEFAULT_MONTHLY_CAPACITY
    daily_control_cap: float = DEFAULT_DAILY_CONTROL_CAP
    depot_lat: float = 0.0
    depot_lon: float = 0.0
    core_ratio: float = 65.0
    core_floor: float = 20.0
    core_step: float = 10.0
    tol_pct: float = 5.0
    tol_mode: str = "target"
    polish_tol_multiplier: float = 1.2
    knn_k: int = 8
    knn_rounds: int = 4
    knn_majority: float = 0.6
    enable_stray_cleanup: bool = True
    enable_majority_vote: bool = True
    enable_swap: bool = True
    swap_rounds: int = 3
    allow_vip_day_move: bool = False
    daily_passes: int = 12

@dataclass
class ZoningResult:
    result_df: pd.DataFrame
    stops_df: pd.DataFrame
    daily_matrix: np.ndarray
    daily_stops: np.ndarray
    final_daily: Dict[str, np.ndarray]
    targets: Dict[str, float]
    loads: Dict[str, float]
    core_ratio_used: float
    metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    infos: List[str] = field(default_factory=list)

# =====================================================================================
#  SECTION 4 — CAPACITY PLANNING & FLEET DIAGNOSTIC
# =====================================================================================
def compute_default_targets(vol_by_truck: Dict[str, float], dissolve: Sequence[str], keep: Sequence[str], new_trucks: Sequence[str], cap_units: float) -> Tuple[Dict[str, float], float]:
    cap_limit = max(0.0, float(cap_units))
    pool = float(sum(float(vol_by_truck.get(t, 0.0)) for t in dissolve))
    targets: Dict[str, float] = {}
    for t in keep:
        vol = max(0.0, float(vol_by_truck.get(t, 0.0)))
        if vol > cap_limit:
            pool += vol - cap_limit
            targets[t] = cap_limit
        else:
            targets[t] = vol
    for t in new_trucks:
        targets[t] = 0.0
    for t in new_trucks:
        if pool <= 1e-6:
            break
        alloc = min(cap_limit - targets.get(t, 0.0), pool)
        targets[t] = targets.get(t, 0.0) + alloc
        pool -= alloc
    for _ in range(60):
        if pool <= 1e-6:
            break
        headroom = {t: max(0.0, cap_limit - tgt) for t, tgt in targets.items() if cap_limit - tgt > 1e-6}
        if not headroom:
            break
        total_head = sum(headroom.values())
        if total_head <= 1e-9:
            break
        ratio = min(1.0, pool / total_head)
        for t, avail in headroom.items():
            alloc = min(pool, avail * ratio)
            targets[t] += alloc
            pool -= alloc
            if pool <= 1e-6:
                pool = 0.0
                break
    return targets, max(0.0, pool)

def diagnose_fleet(df: pd.DataFrame, cfg: ZoningConfig) -> pd.DataFrame:
    output_rows = []
    if df.empty:
        return pd.DataFrame()
    for truck, group in df.groupby(cfg.truck_col, dropna=False):
        truck_id = str(truck).strip()
        if truck_id.lower() in NO_TRUCK_TOKENS:
            continue
        daily_vol = np.zeros(WORKING_DAYS, dtype=float)
        daily_stops = np.zeros(WORKING_DAYS, dtype=float)
        for _, row in group.iterrows():
            days, _ = parse_days_from_string(row.get(cfg.day_col))
            if not days:
                days = list(range(WORKING_DAYS))
            vol_m = float(pd.to_numeric(row.get(cfg.vol_col), errors="coerce") or 0.0)
            vol_per_day = vol_m / max(1, len(days)) / WEEKS_PER_MONTH
            for d in days:
                daily_vol[d] += vol_per_day
                daily_stops[d] += 1
        tot_vol = float(pd.to_numeric(group[cfg.vol_col], errors="coerce").fillna(0).sum())
        peak_v = float(daily_vol.max())
        peak_s = int(round(float(daily_stops.max())))
        if peak_v > cfg.daily_control_cap:
            status = "🔴 เกินเพดาน"
        elif peak_v >= OPTIMAL_MIN:
            status = "🟢 เหมาะสม"
        elif peak_v >= AVOID_MIN:
            status = "🟡 ควรเลี่ยง"
        else:
            status = "⚪ เบาเกิน"
        output_rows.append({
            "เบอร์รถ": truck_id,
            "จำนวนลูกค้า": int(len(group)),
            "ยอด/เดือน": int(round(tot_vol)),
            "ภาระงาน(%)": round(tot_vol / max(1.0, cfg.monthly_capacity) * 100.0, 1),
            "โหลดสูงสุด/วัน": int(round(peak_v)),
            "จุดจอด": peak_s,
            "ส่วนเกิน/วัน": int(round(max(0.0, peak_v - cfg.daily_control_cap))),
            "สถานะ": status,
        })
    out_df = pd.DataFrame(output_rows)
    return out_df.sort_values("โหลดสูงสุด/วัน", ascending=False).reset_index(drop=True) if not out_df.empty else out_df

# =====================================================================================
#  SECTION 5 — CORE ZONING & SATELLITE POCKET CONSOLIDATION
# =====================================================================================
def _tolerance_for(truck: str, targets: Dict[str, float], cfg: ZoningConfig) -> float:
    if cfg.tol_mode == "target":
        return max(40.0, (cfg.tol_pct / 100.0) * max(0.0, targets.get(truck, 0.0)))
    return (cfg.tol_pct / 100.0) * max(0.0, cfg.monthly_capacity)

def build_stops(df: pd.DataFrame, cfg: ZoningConfig) -> pd.DataFrame:
    stops = (
        df.groupby("coord_key", dropna=False)
        .agg(
            lat=(cfg.lat_col, "first"),
            lon=(cfg.lon_col, "first"),
            x=("x", "first"),
            y=("y", "first"),
            total_vol=(cfg.vol_col, "sum"),
            n_cust=(cfg.id_col, "count"),
            orig_truck=(cfg.truck_col, "first"),
            has_vip_lock=("is_vip_locked", "any"),
        )
        .reset_index()
    )
    stops["orig_truck"] = stops["orig_truck"].astype(str).str.strip()
    stops["total_vol"] = pd.to_numeric(stops["total_vol"], errors="coerce").fillna(0.0).astype(float)
    return stops

def compute_core_keys(stops: pd.DataFrame, ratio: float, eligible_trucks: Sequence[str], ratio_override: Optional[Dict[str, float]] = None) -> set:
    core_keys = set()
    if stops.empty or ratio <= 0:
        return core_keys
    ratio_override = ratio_override or {}
    for truck in eligible_trucks:
        t_ratio = min(float(ratio), float(ratio_override.get(truck, 100.0)))
        if t_ratio <= 0:
            continue
        group = stops[stops["orig_truck"] == truck].copy()
        if group.empty:
            continue
        weights = np.maximum(group["total_vol"].to_numpy(dtype=float), 1e-9)
        cx, cy = float(np.average(group["x"], weights=weights)), float(np.average(group["y"], weights=weights))
        group["_dist_sq"] = (group["x"] - cx) ** 2 + (group["y"] - cy) ** 2
        group = group.sort_values("_dist_sq", ascending=True)
        tot = max(1e-6, float(group["total_vol"].sum()))
        selected = (group["total_vol"].cumsum() / tot) <= (t_ratio / 100.0)
        if not selected.any() and len(group):
            selected.iloc[0] = True
        core_keys.update(group.loc[selected, "coord_key"].tolist())
    return core_keys

def _assign_capacitated(stops: pd.DataFrame, trucks: List[str], targets: Dict[str, float], tolerance: Dict[str, float], seeds: Dict[str, Tuple[float, float]], max_rounds: int = 80) -> Tuple[np.ndarray, Dict[str, float]]:
    n_stops, n_trucks = len(stops), len(trucks)
    if n_stops == 0:
        return np.asarray([], dtype=int), {t: 0.0 for t in trucks}
    if n_trucks == 0:
        return np.full(n_stops, -1, dtype=int), {}
    coords = stops[["x", "y"]].to_numpy(dtype=float)
    vols = stops["total_vol"].to_numpy(dtype=float)
    t_idx = {t: i for i, t in enumerate(trucks)}
    assigned = np.array([t_idx.get(v, -1) if isinstance(v, str) else -1 for v in stops["assigned_truck"].tolist()], dtype=int)
    loads = {t: 0.0 for t in trucks}
    for idx in np.where(assigned >= 0)[0]:
        loads[trucks[assigned[idx]]] += vols[idx]
    for _ in range(max(1, int(max_rounds))):
        el_trucks = [t for t in trucks if loads[t] < targets.get(t, 0.0) - 1e-9]
        if not el_trucks:
            break
        cand_indices = np.where(assigned < 0)[0]
        if cand_indices.size == 0:
            break
        centers = []
        for t in el_trucks:
            mask = assigned == t_idx[t]
            if mask.any():
                center = np.average(coords[mask], axis=0, weights=np.maximum(vols[mask], 1e-9))
            else:
                center = np.asarray(seeds.get(t, (coords[:, 0].mean(), coords[:, 1].mean())), dtype=float)
            centers.append(center)
        centers_arr = np.asarray(centers, dtype=float)
        dist = np.sqrt(((coords[cand_indices][:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2))
        caps = np.asarray([targets.get(t, 0.0) + tolerance.get(t, 0.0) for t in el_trucks], dtype=float)
        cur_loads = np.asarray([loads[t] for t in el_trucks], dtype=float)
        feasible = cur_loads[None, :] + vols[cand_indices][:, None] <= caps[None, :]
        dist = np.where(feasible, dist, np.inf)
        best_t_local = dist.argmin(axis=1)
        best_d = dist[np.arange(len(cand_indices)), best_t_local]
        feas_rows = np.isfinite(best_d)
        if not feas_rows.any():
            break
        feas_cand = cand_indices[feas_rows]
        feas_t_indices = best_t_local[feas_rows]
        order = np.argsort(best_d[feas_rows])
        placed = False
        for pos in order:
            s_idx = feas_cand[pos]
            if assigned[s_idx] >= 0:
                continue
            t = el_trucks[feas_t_indices[pos]]
            if loads[t] + vols[s_idx] > targets.get(t, 0.0) + tolerance.get(t, 0.0):
                continue
            assigned[s_idx] = t_idx[t]
            loads[t] += vols[s_idx]
            placed = True
        if not placed:
            break
    for s_idx in np.where(assigned < 0)[0]:
        cand_trucks = [t for t in trucks if targets.get(t, 0.0) > 0]
        if not cand_trucks:
            continue
        best_t = max(cand_trucks, key=lambda t: targets.get(t, 0.0) - loads.get(t, 0.0))
        tgt = targets.get(best_t, 0.0)
        if loads.get(best_t, 0.0) + vols[s_idx] <= tgt + max(tolerance.get(best_t, 0.0), tgt * 0.20):
            assigned[s_idx] = t_idx[best_t]
            loads[best_t] += vols[s_idx]
    return assigned, loads

def consolidate_satellite_pockets(stops: pd.DataFrame, trucks: List[str], targets: Dict[str, float], loads: Dict[str, float], tolerance: Dict[str, float], pocket_radius_m: float = 450.0) -> Tuple[pd.DataFrame, Dict[str, float]]:
    stops = stops.copy()
    loads = dict(loads)
    n = len(stops)
    if n <= 1:
        return stops, loads
    coords = stops[["x", "y"]].to_numpy(dtype=float)
    if HAS_SCIPY:
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=pocket_radius_m)
    else:
        pairs = []
        for i in range(n):
            dists_sq = ((coords[i] - coords[i+1:]) ** 2).sum(axis=1)
            for j_offset, d2 in enumerate(dists_sq):
                if d2 <= pocket_radius_m ** 2:
                    pairs.append((i, i + 1 + j_offset))
    parent = list(range(n))
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for p in path:
            parent[p] = i
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    for i, j in pairs:
        union(i, j)
    components: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        components.setdefault(root, []).append(i)
    centers = {}
    for t in trucks:
        grp = stops[stops["assigned_truck"] == t]
        if not grp.empty:
            w = np.maximum(grp["total_vol"].to_numpy(dtype=float), 1e-9)
            centers[t] = np.average(grp[["x", "y"]].to_numpy(dtype=float), axis=0, weights=w)
    for root, member_indices in components.items():
        if len(member_indices) >= n * 0.45:
            continue
        member_stops = stops.iloc[member_indices]
        assigned_trucks = [t for t in member_stops["assigned_truck"].unique().tolist() if t in trucks]
        if len(set(assigned_trucks)) > 1:
            pw = np.maximum(member_stops["total_vol"].to_numpy(dtype=float), 1e-9)
            pocket_center = np.average(member_stops[["x", "y"]].to_numpy(dtype=float), axis=0, weights=pw)
            vip_trucks = member_stops.loc[member_stops["is_locked"].astype(bool), "assigned_truck"].tolist()
            vip_trucks = [t for t in vip_trucks if t in trucks]
            best_truck = vip_trucks[0] if vip_trucks else None
            if best_truck is None:
                best_dist = float("inf")
                for t in assigned_trucks:
                    if t in centers:
                        d = np.sqrt(((pocket_center - centers[t]) ** 2).sum())
                        if d < best_dist:
                            best_dist = d
                            best_truck = t
            if best_truck is None and assigned_trucks:
                best_truck = assigned_trucks[0]
            if best_truck is not None:
                for s_idx in member_indices:
                    old_t = stops.at[s_idx, "assigned_truck"]
                    if old_t != best_truck:
                        if not bool(stops.at[s_idx, "is_locked"]) or old_t not in trucks:
                            v = float(stops.at[s_idx, "total_vol"])
                            stops.at[s_idx, "assigned_truck"] = best_truck
                            if old_t in loads:
                                loads[old_t] -= v
                            loads[best_truck] = loads.get(best_truck, 0.0) + v
    return stops, loads

def cleanup_stray_points(stops: pd.DataFrame, trucks: List[str], targets: Dict[str, float], loads: Dict[str, float], tolerance: Dict[str, float], multiplier: float, rounds: int = 3) -> Tuple[pd.DataFrame, Dict[str, float]]:
    stops = stops.copy()
    loads = dict(loads)
    for _ in range(max(1, int(rounds))):
        changed = False
        centers = {}
        for t in trucks:
            grp = stops[stops["assigned_truck"] == t]
            if not grp.empty:
                w = np.maximum(grp["total_vol"].to_numpy(dtype=float), 1e-9)
                centers[t] = (float(np.average(grp["x"], weights=w)), float(np.average(grp["y"], weights=w)))
        for t in trucks:
            mask = (stops["assigned_truck"] == t) & (~stops["is_locked"].astype(bool))
            grp = stops[mask]
            if len(grp) < 3 or t not in centers:
                continue
            cx, cy = centers[t]
            dists = np.sqrt((grp["x"] - cx) ** 2 + (grp["y"] - cy) ** 2)
            stray_indices = grp.index[dists > (float(dists.median()) * 3.0 + 1.0)]
            for s_idx in stray_indices:
                stop = stops.loc[s_idx]
                own_d_sq = (stop["x"] - cx) ** 2 + (stop["y"] - cy) ** 2
                best_t, best_d_sq = None, float("inf")
                for ot, (ox, oy) in centers.items():
                    if ot == t:
                        continue
                    d_sq = (stop["x"] - ox) ** 2 + (stop["y"] - oy) ** 2
                    if d_sq < best_d_sq:
                        best_d_sq, best_t = d_sq, ot
                if best_t is None:
                    continue
                proj = loads.get(best_t, 0.0) + float(stop["total_vol"])
                allowed = targets.get(best_t, 0.0) + tolerance.get(best_t, 0.0) * multiplier
                if best_d_sq < own_d_sq * 0.6 and proj <= allowed:
                    vol = float(stop["total_vol"])
                    stops.at[s_idx, "assigned_truck"] = best_t
                    loads[t] -= vol
                    loads[best_t] += vol
                    changed = True
        if not changed:
            break
    return stops, loads

def majority_vote_smoothing(stops: pd.DataFrame, trucks: List[str], targets: Dict[str, float], loads: Dict[str, float], tolerance: Dict[str, float], cfg: ZoningConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    stops = stops.copy().reset_index(drop=True)
    loads = dict(loads)
    n_stops = len(stops)
    if n_stops < cfg.knn_k + 2:
        return stops, loads
    coords = stops[["x", "y"]].to_numpy(dtype=float)
    nbrs = knn_indices(coords, cfg.knn_k)
    locked = stops["is_locked"].astype(bool).to_numpy()
    vols = stops["total_vol"].to_numpy(dtype=float)
    val_trucks = set(trucks)
    for _ in range(max(1, int(cfg.knn_rounds))):
        changed = False
        assigned = stops["assigned_truck"].astype(object).to_numpy().copy()
        for idx in range(n_stops):
            if locked[idx]:
                continue
            cur_t = assigned[idx]
            nbr_t = [v for v in assigned[nbrs[idx]].tolist() if v in val_trucks]
            if not nbr_t:
                continue
            vals, counts = np.unique(np.asarray(nbr_t, dtype=str), return_counts=True)
            maj_t = str(vals[np.argmax(counts)])
            if maj_t == cur_t or int(counts.max()) < cfg.knn_k * cfg.knn_majority:
                continue
            proj = loads.get(maj_t, 0.0) + vols[idx]
            if proj > targets.get(maj_t, 0.0) + tolerance.get(maj_t, 0.0) * cfg.polish_tol_multiplier:
                continue
            if cur_t in val_trucks:
                loads[cur_t] -= vols[idx]
            loads[maj_t] = proj
            assigned[idx] = maj_t
            changed = True
        stops["assigned_truck"] = assigned
        if not changed:
            break
    return stops, loads

def swap_improve(stops: pd.DataFrame, trucks: List[str], targets: Dict[str, float], loads: Dict[str, float], tolerance: Dict[str, float], rounds: int = 3, max_pairs: int = 60) -> Tuple[pd.DataFrame, Dict[str, float]]:
    stops = stops.copy()
    loads = dict(loads)
    for _ in range(max(1, int(rounds))):
        changed = False
        centers = {}
        for t in trucks:
            grp = stops[stops["assigned_truck"] == t]
            if not grp.empty:
                w = np.maximum(grp["total_vol"].to_numpy(dtype=float), 1e-9)
                centers[t] = np.asarray([np.average(grp["x"], weights=w), np.average(grp["y"], weights=w)], dtype=float)
        for ta, tb in combinations([t for t in trucks if t in centers], 2):
            ca, cb = centers[ta], centers[tb]
            ga = stops[(stops["assigned_truck"] == ta) & (~stops["is_locked"].astype(bool))]
            gb = stops[(stops["assigned_truck"] == tb) & (~stops["is_locked"].astype(bool))]
            if ga.empty or gb.empty:
                continue
            co_a, co_b = ga[["x", "y"]].to_numpy(dtype=float), gb[["x", "y"]].to_numpy(dtype=float)
            gain_a = np.sqrt(((co_a - ca) ** 2).sum(axis=1)) - np.sqrt(((co_a - cb) ** 2).sum(axis=1))
            gain_b = np.sqrt(((co_b - cb) ** 2).sum(axis=1)) - np.sqrt(((co_b - ca) ** 2).sum(axis=1))
            c_a = ga.index[np.argsort(-gain_a)][:max_pairs]
            c_b = gb.index[np.argsort(-gain_b)][:max_pairs]
            gl_a, gl_b = dict(zip(ga.index, gain_a)), dict(zip(gb.index, gain_b))
            used_b = set()
            for ia in c_a:
                if gl_a[ia] <= 0:
                    break
                for ib in c_b:
                    if ib in used_b or gl_b[ib] <= 0 or gl_a[ia] + gl_b[ib] <= 0:
                        continue
                    va, vb = float(stops.at[ia, "total_vol"]), float(stops.at[ib, "total_vol"])
                    n_la, n_lb = loads.get(ta, 0.0) - va + vb, loads.get(tb, 0.0) - vb + va
                    if n_la > targets.get(ta, 0.0) + tolerance.get(ta, 0.0) or n_lb > targets.get(tb, 0.0) + tolerance.get(tb, 0.0):
                        continue
                    stops.at[ia, "assigned_truck"], stops.at[ib, "assigned_truck"] = tb, ta
                    loads[ta], loads[tb] = n_la, n_lb
                    used_b.add(ib)
                    changed = True
                    break
        if not changed:
            break
    return stops, loads

# =====================================================================================
#  SECTION 6 — DAILY LOAD SMOOTHING & DEPOT HUB DISTANCE
# =====================================================================================
def smooth_daily_loads(opt: pd.DataFrame, cfg: ZoningConfig, trucks: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    opt = opt.copy().reset_index(drop=True)
    trucks = list(dict.fromkeys(str(t).strip() for t in trucks))
    n_rows = len(opt)
    if n_rows == 0:
        empty_mat = np.zeros((0, WORKING_DAYS), dtype=float)
        return opt, empty_mat, empty_mat.copy(), {t: np.zeros(WORKING_DAYS, dtype=float) for t in trucks}
    if "สถานะการย้ายวัน" not in opt.columns:
        opt["สถานะการย้ายวัน"] = "-"
    if "is_vip_locked" not in opt.columns:
        opt["is_vip_locked"] = False
    m_vols = pd.to_numeric(opt[cfg.vol_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    day_map: Dict[int, List[int]] = {}
    for idx in opt.index:
        days, status = parse_days_from_string(opt.at[idx, cfg.day_col])
        if not days:
            days = list(range(WORKING_DAYS))
        day_map[idx] = list(sorted(set(days)))
        opt.at[idx, "_day_status"] = status
    def recompute_daily():
        d_vol = {t: np.zeros(WORKING_DAYS, dtype=float) for t in trucks}
        d_stops = {t: np.zeros(WORKING_DAYS, dtype=float) for t in trucks}
        for idx in opt.index:
            t = str(opt.at[idx, "เบอร์รถใหม่"]).strip()
            if t not in d_vol:
                continue
            days = day_map[idx]
            if not days:
                continue
            v_day = m_vols[idx] / len(days) / WEEKS_PER_MONTH
            for d in days:
                if 0 <= d < WORKING_DAYS:
                    d_vol[t][d] += v_day
                    d_stops[t][d] += 1
        return d_vol, d_stops
    d_cap = max(1.0, float(cfg.daily_control_cap))
    for _ in range(max(1, int(cfg.daily_passes))):
        d_vol, d_stops = recompute_daily()
        changed = False
        for t in trucks:
            if t not in d_vol:
                continue
            score = d_vol[t] / d_cap
            src_order = np.argsort(-score)
            for src in src_order:
                src = int(src)
                if not (d_vol[t][src] > d_cap):
                    continue
                dest_score = (d_vol[t] / d_cap).copy()
                dest_score[src] = np.inf
                dest_candidates = np.argsort(dest_score)
                dest = None
                for cand in dest_candidates:
                    cand = int(cand)
                    if not np.isfinite(dest_score[cand]) or d_vol[t][cand] >= d_cap:
                        continue
                    dest = cand
                    break
                if dest is None:
                    continue
                movable = []
                for idx in opt.index:
                    if str(opt.at[idx, "เบอร์รถใหม่"]).strip() != t:
                        continue
                    if not cfg.allow_vip_day_move and bool(opt.at[idx, "is_vip_locked"]):
                        continue
                    c_days = day_map[idx]
                    if src not in c_days or dest in c_days or len(c_days) > 3:
                        continue
                    movable.append(idx)
                if not movable:
                    continue
                movable.sort(key=lambda idx: -m_vols[idx] / max(1, len(day_map[idx])))
                excess_v = max(0.0, d_vol[t][src] - TARGET_DAY_CAP)
                shifted_v = 0.0
                for idx in movable:
                    c_days = day_map[idx]
                    v_day = m_vols[idx] / max(1, len(c_days)) / WEEKS_PER_MONTH
                    if d_vol[t][dest] + v_day > d_cap:
                        continue
                    new_days = sorted(set(dest if d == src else d for d in c_days))
                    if len(new_days) != len(c_days):
                        continue
                    day_map[idx] = new_days
                    note = f"{DAY_SHORT[src]}→{DAY_SHORT[dest]}"
                    prev = str(opt.at[idx, "สถานะการย้ายวัน"]).strip()
                    if not prev or prev.lower() in NO_TRUCK_TOKENS:
                        opt.at[idx, "สถานะการย้ายวัน"] = note
                    else:
                        notes = [n.strip() for n in prev.split(",") if n.strip()]
                        if note not in notes:
                            notes.append(note)
                        opt.at[idx, "สถานะการย้ายวัน"] = ", ".join(notes)
                    d_vol[t][src] -= v_day
                    d_vol[t][dest] += v_day
                    d_stops[t][src] -= 1
                    d_stops[t][dest] += 1
                    shifted_v += v_day
                    changed = True
                    if d_vol[t][src] <= d_cap and shifted_v >= excess_v:
                        break
        if not changed:
            break
    for idx in opt.index:
        opt.at[idx, "วันจัดส่ง(ใหม่)"] = format_days_to_string(day_map[idx])
        opt.at[idx, "วันจัดส่ง(ย่อ)"] = format_days_short(day_map[idx])
    daily_mat = np.zeros((n_rows, WORKING_DAYS), dtype=float)
    daily_s_mat = np.zeros((n_rows, WORKING_DAYS), dtype=float)
    for idx in opt.index:
        days = day_map[idx]
        if not days:
            continue
        v_day = m_vols[idx] / len(days) / WEEKS_PER_MONTH
        for d in days:
            daily_mat[idx, d] = v_day
            daily_s_mat[idx, d] = 1.0
    final_daily, _ = recompute_daily()
    return opt, daily_mat, daily_s_mat, final_daily

def compute_depot_hub_distance(group: pd.DataFrame, depot_lat: float, depot_lon: float) -> float:
    if group.empty or depot_lat == 0.0 or depot_lon == 0.0:
        return 0.0
    dx, dy, _ = project_xy([depot_lat] + group["_lat_clean"].tolist(), [depot_lon] + group["_lon_clean"].tolist())
    depot_x, depot_y = dx[0], dy[0]
    cust_x, cust_y = dx[1:], dy[1:]
    if len(cust_x) == 0:
        return 0.0
    weights = np.ones(len(group), dtype=float)
    if "total_vol" in group.columns:
        weights = np.maximum(pd.to_numeric(group["total_vol"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 1e-9)
    distances = np.sqrt((cust_x - depot_x) ** 2 + (cust_y - depot_y) ** 2) * 2.0
    return float(np.average(distances, weights=weights))

def calculate_peak_daily_loads(data: pd.DataFrame, truck_col: str, vol_col: str, day_col: str) -> Dict[str, float]:
    output = {}
    if data.empty:
        return output
    for truck, group in data.groupby(truck_col, dropna=False):
        t_id = str(truck).strip()
        if t_id.lower() in NO_TRUCK_TOKENS:
            continue
        daily = np.zeros(WORKING_DAYS, dtype=float)
        for _, row in group.iterrows():
            days, _ = parse_days_from_string(row.get(day_col))
            if not days:
                days = list(range(WORKING_DAYS))
            vol_m = float(pd.to_numeric(row.get(vol_col), errors="coerce") or 0.0)
            v_day = vol_m / len(days) / WEEKS_PER_MONTH
            for d in days:
                daily[d] += v_day
        output[t_id] = float(daily.max())
    return output

# =====================================================================================
#  SECTION 7 — ENTERPRISE UI & MODERN STYLING (SAFE INJECTION)
# =====================================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Sarabun:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sarabun', 'Plus+Jakarta+Sans', sans-serif !important;
}

.stApp {
    background: #F8FAFC !important;
}

/* Sidebar Styling & Text Visibility */
[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B !important;
}
[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] span {
    color: #38BDF8 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #38BDF8 !important;
    border-bottom: 1px solid #1E293B !important;
    padding-bottom: 8px !important;
}

/* Fix Dropdown & Input Text inside Sidebar */
[data-testid="stSidebar"] input, [data-testid="stSidebar"] select, [data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #0F172A !important;
}

/* Modern Tabs Navigation */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #E2E8F0;
    padding: 6px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    background-color: transparent;
    border-radius: 8px;
    color: #475569;
    font-weight: 700;
    padding: 0 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #0284C7 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

/* Cards & Containers */
.metric-card-modern {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s;
    margin-bottom: 12px;
}
.metric-card-modern:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.6rem 1.2rem;
    box-shadow: 0 4px 12px rgba(2,132,199,0.25);
    transition: all 0.2s;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #0369A1 0%, #075985 100%) !important;
    box-shadow: 0 6px 16px rgba(2,132,199,0.35);
}

/* Custom Table Container */
.custom-table-container {
    overflow-x: auto;
    background: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);
    margin-bottom: 20px;
}
.custom-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
    text-align: center;
}
.custom-table th {
    background: #F1F5F9;
    color: #334155;
    font-weight: 700;
    padding: 12px 10px;
    border-bottom: 2px solid #CBD5E1;
}
.custom-table td {
    padding: 10px;
    border-bottom: 1px solid #E2E8F0;
    color: #1E293B;
    font-weight: 500;
}
.custom-table tr:hover td {
    background: #F8FAFC;
}
.text-danger {
    color: #DC2626 !important;
    font-weight: 800 !important;
}
.text-primary {
    color: #0284C7 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

def render_enterprise_metric(label: str, value: str, delta: str = "", status: str = "normal"):
    border_colors = {"good": "#10B981", "warning": "#F59E0B", "danger": "#EF4444", "normal": "#0284C7"}
    bg_delta = {"good": "#ECFDF5", "warning": "#FFFBEB", "danger": "#FEF2F2", "normal": "#F0F9FF"}
    txt_delta = {"good": "#059669", "warning": "#D97706", "danger": "#DC2626", "normal": "#0284C7"}
    
    bc = border_colors.get(status, "#0284C7")
    dc = bg_delta.get(status, "#F0F9FF")
    tc = txt_delta.get(status, "#0284C7")
    
    delta_html = ""
    if delta:
        delta_html = f'''<div style="margin-top: 8px;"><span style="background:{dc}; color:{tc}; padding:3px 10px; border-radius:6px; font-weight:700; font-size:0.85rem; display:inline-block;">{delta}</span></div>'''
        
    html = f'''
    <div class="metric-card-modern" style="border-left: 5px solid {bc};">
        <div style="color: #64748B; font-size: 0.9rem; font-weight: 600; margin-bottom: 4px;">{label}</div>
        <div style="color: #0F172A; font-size: 1.8rem; font-weight: 800; line-height: 1.2;">{value}</div>
        {delta_html}
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

def reset_results():
    for k in ('result', 'zoning_cfg_used'):
        st.session_state.pop(k, None)

def show_loader(placeholder, msg: str):
    html = f'<div style="text-align:center; padding:2rem; color:#0284C7; font-weight:bold; font-size:1.2rem; border-radius:12px; background:#FFFFFF; border:1px solid #E2E8F0;">{msg}</div>'
    placeholder.markdown(html, unsafe_allow_html=True)

# =====================================================================================
#  SECTION 8 — MAIN ENGINE EXECUTION
# =====================================================================================
def run_multi_donor_zoning(df: pd.DataFrame, cfg: ZoningConfig, target_pcts: Dict[str, float], dissolve_trucks: Sequence[str], relieve_trucks: Sequence[str], new_trucks: Sequence[str], manual_locks: Sequence[str], exclude_trucks: Sequence[str] = None) -> ZoningResult:
    warnings, infos = [], []
    exclude_trucks = exclude_trucks or []
    if df is None or df.empty:
        raise ValueError("ไม่มีข้อมูลสำหรับประมวลผล")
    opt = df.copy().reset_index(drop=True)
    
    opt["_lat_clean"] = pd.to_numeric(opt["_lat_clean"], errors="coerce")
    opt["_lon_clean"] = pd.to_numeric(opt["_lon_clean"], errors="coerce")
    opt[cfg.vol_col] = pd.to_numeric(opt[cfg.vol_col], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    opt[cfg.truck_col] = opt[cfg.truck_col].astype(str).str.strip()
    opt[cfg.id_col] = opt[cfg.id_col].astype(str).str.strip()

    valid_mask = opt["_lat_clean"].notna() & opt["_lon_clean"].notna() & (~opt[cfg.truck_col].str.lower().isin(NO_TRUCK_TOKENS))
    opt = opt[valid_mask].reset_index(drop=True)
    opt["coord_key"] = opt["_lat_clean"].round(5).astype(str) + "," + opt["_lon_clean"].round(5).astype(str)
    px, py, _ = project_xy(opt["_lat_clean"].to_numpy(dtype=float), opt["_lon_clean"].to_numpy(dtype=float), lat0=cfg.depot_lat if cfg.depot_lat != 0 else None)
    opt["x"], opt["y"] = px, py

    vip_status = opt["VIP_Status"].astype(str).str.upper().str.strip() if "VIP_Status" in opt.columns else pd.Series(["ปกติ"] * len(opt))
    locked_ids = {str(v).strip() for v in manual_locks if str(v).strip()}
    opt["is_vip_locked"] = vip_status.eq("VIP") | opt[cfg.id_col].isin(locked_ids)
    opt["สถานะการย้ายวัน"] = "-"

    exclude_set = set(exclude_trucks)
    excluded_mask = opt[cfg.truck_col].isin(exclude_set)
    excluded_df = opt[excluded_mask].copy()
    opt = opt[~excluded_mask].reset_index(drop=True)

    if opt.empty and not excluded_df.empty:
        raise ValueError("รถทั้งหมดถูกล็อกไว้ ไม่เหลือข้อมูลสำหรับประมวลผลจัดสาย")
    elif opt.empty:
        raise ValueError("ไม่มีข้อมูลสำหรับประมวลผล")

    all_orig_trucks = clean_truck_ids(opt[cfg.truck_col].unique())
    dissolve = [t for t in all_orig_trucks if t in set(dissolve_trucks)]
    kept = [t for t in all_orig_trucks if t not in set(dissolve_trucks)]
    norm_new = [t.strip() for t in new_trucks if t.strip() and t.strip() not in kept and t.strip().lower() not in NO_TRUCK_TOKENS]
    active = kept + list(dict.fromkeys(norm_new))
    if not active:
        raise ValueError("ไม่เหลือรถที่สามารถรับการจัดสรรงานได้")

    targets = {t: max(0.0, cfg.monthly_capacity * float(target_pcts.get(t, 0.0)) / 100.0) for t in active}
    tolerance = {t: _tolerance_for(t, targets, cfg) for t in active}
    
    opt[cfg.lat_col] = opt["_lat_clean"]
    opt[cfg.lon_col] = opt["_lon_clean"]
    stops = build_stops(opt, cfg)

    vol_by_orig = stops.groupby("orig_truck")["total_vol"].sum().to_dict()
    ratio_override = {}
    for t in relieve_trucks:
        c_vol = float(vol_by_orig.get(t, 0.0))
        if c_vol > 0 and t in targets:
            ratio_override[t] = max(0.0, min(100.0, targets[t] / c_vol * 100.0 * 0.92))

    core_eligible = [t for t in kept if t not in set(dissolve)]
    seeds: Dict[str, Tuple[float, float]] = {}
    for t in kept:
        grp = stops[stops["orig_truck"] == t]
        if not grp.empty:
            w = np.maximum(grp["total_vol"].to_numpy(dtype=float), 1e-9)
            seeds[t] = (float(np.average(grp["x"], weights=w)), float(np.average(grp["y"], weights=w)))

    gw = np.maximum(stops["total_vol"].to_numpy(dtype=float), 1e-9)
    gx, gy = float(np.average(stops["x"], weights=gw)), float(np.average(stops["y"], weights=gw))
    if cfg.depot_lat != 0 and cfg.depot_lon != 0:
        dx_depot, dy_depot, _ = project_xy([cfg.depot_lat], [cfg.depot_lon], lat0=cfg.depot_lat)
        depot_coord = (float(dx_depot[0]), float(dy_depot[0]))
    else:
        depot_coord = (gx, gy)

    if norm_new:
        init_cores = compute_core_keys(stops, cfg.core_ratio, core_eligible, ratio_override)
        spool = stops[~stops["coord_key"].isin(init_cores)]
        if spool.empty:
            spool = stops
        new_centers = kmeans_seeds(spool[["x", "y"]].to_numpy(dtype=float), spool["total_vol"].to_numpy(dtype=float), len(norm_new))
        if len(new_centers) == 0:
            new_centers = np.asarray([depot_coord], dtype=float)
        for idx, t in enumerate(norm_new):
            c_idx = min(idx, len(new_centers) - 1)
            seeds[t] = (float(new_centers[c_idx, 0]), float(new_centers[c_idx, 1]))

    def run_single_pass(core_ratio: float):
        c_stops = stops.copy()
        c_keys = compute_core_keys(c_stops, core_ratio, core_eligible, ratio_override)
        c_stops["is_core_locked"] = c_stops["coord_key"].isin(c_keys)
        c_stops["is_locked"] = c_stops["has_vip_lock"].astype(bool) | c_stops["is_core_locked"].astype(bool)
        c_stops["assigned_truck"] = None
        for s_idx, row in c_stops.iterrows():
            ot = str(row["orig_truck"]).strip()
            if bool(row["is_locked"]) and ot in active:
                c_stops.at[s_idx, "assigned_truck"] = ot
        assignment, pass_loads = _assign_capacitated(c_stops, active, targets, tolerance, seeds)
        c_stops["assigned_truck"] = [active[ti] if ti >= 0 else OVERFLOW_LABEL for ti in assignment]
        c_stops.loc[c_stops["is_locked"] & (~c_stops["orig_truck"].isin(active)), "is_locked"] = False
        return c_stops, pass_loads

    cur_ratio = max(0.0, min(100.0, float(cfg.core_ratio)))
    min_ratio = max(0.0, min(cur_ratio, float(cfg.core_floor)))
    c_step = max(1.0, float(cfg.core_step))
    best_result = None

    while True:
        c_stops, c_loads = run_single_pass(cur_ratio)
        tot_viol = sum(max(0.0, abs(c_loads.get(t, 0.0) - targets.get(t, 0.0)) - tolerance.get(t, 0.0)) for t in active if targets.get(t, 0.0) > 0)
        over_v = float(c_stops.loc[c_stops["assigned_truck"] == OVERFLOW_LABEL, "total_vol"].sum())
        score = tot_viol + over_v * 2.0
        if best_result is None or score < best_result[3]:
            best_result = (c_stops, c_loads, cur_ratio, score)
        if score <= 1e-6 or cur_ratio <= min_ratio:
            break
        cur_ratio = max(min_ratio, cur_ratio - c_step)

    assigned_stops, loads, ratio_used, _ = best_result
    assigned_stops, loads = consolidate_satellite_pockets(assigned_stops, active, targets, loads, tolerance, pocket_radius_m=450.0)
    if cfg.enable_stray_cleanup:
        assigned_stops, loads = cleanup_stray_points(assigned_stops, active, targets, loads, tolerance, cfg.polish_tol_multiplier)
    if cfg.enable_majority_vote:
        assigned_stops, loads = majority_vote_smoothing(assigned_stops, active, targets, loads, tolerance, cfg)
    if cfg.enable_swap:
        assigned_stops, loads = swap_improve(assigned_stops, active, targets, loads, tolerance, cfg.swap_rounds)
    assigned_stops, loads = consolidate_satellite_pockets(assigned_stops, active, targets, loads, tolerance, pocket_radius_m=450.0)

    truck_mapping = dict(zip(assigned_stops["coord_key"], assigned_stops["assigned_truck"]))
    core_mapping = dict(zip(assigned_stops["coord_key"], assigned_stops["is_core_locked"]))
    opt["เบอร์รถใหม่"] = opt["coord_key"].map(truck_mapping).fillna(OVERFLOW_LABEL).astype(str)
    opt["is_core_locked"] = opt["coord_key"].map(core_mapping).fillna(False).astype(bool)
    opt["is_locked"] = opt["is_vip_locked"].astype(bool) | opt["is_core_locked"].astype(bool)

    orig_s = opt[cfg.truck_col].astype(str).str.strip()
    new_s = opt["เบอร์รถใหม่"].astype(str).str.strip()
    opt["สถานะ"] = np.where(orig_s == new_s, "คงเดิม", "ย้ายไปสาย " + new_s)
    opt.loc[opt["is_locked"] & (orig_s == new_s), "สถานะ"] = "คงเดิม 🔒"

    if not excluded_df.empty:
        excluded_df["เบอร์รถใหม่"] = excluded_df[cfg.truck_col]
        excluded_df["is_core_locked"] = True
        excluded_df["is_vip_locked"] = True
        excluded_df["is_locked"] = True
        excluded_df["สถานะ"] = "คงเดิม (ล็อกคัน) 🔒"
        opt = pd.concat([opt, excluded_df], ignore_index=True)

    all_final_trucks = active + list(exclude_set)
    opt, daily_matrix, daily_stops_matrix, final_daily = smooth_daily_loads(opt, cfg, all_final_trucks)

    comp_before = [compute_depot_hub_distance(grp, cfg.depot_lat, cfg.depot_lon) for _, grp in opt.groupby(cfg.truck_col)]
    comp_after = [compute_depot_hub_distance(grp, cfg.depot_lat, cfg.depot_lon) for t, grp in opt.groupby("เบอร์รถใหม่") if t != OVERFLOW_LABEL]
    peak_before = calculate_peak_daily_loads(opt, cfg.truck_col, cfg.vol_col, cfg.day_col)
    peak_after = {t: float(v.max()) for t, v in final_daily.items()}
    moved_m_full = opt[cfg.truck_col].astype(str).str.strip() != opt["เบอร์รถใหม่"].astype(str).str.strip()

    metrics = {
        "compact_before_km": float(np.mean(comp_before)) / 1000.0 if comp_before else 0.0,
        "compact_after_km": float(np.mean(comp_after)) / 1000.0 if comp_after else 0.0,
        "over_before": int(sum(1 for v in peak_before.values() if v > cfg.daily_control_cap)),
        "over_after": int(sum(1 for v in peak_after.values() if v > cfg.daily_control_cap)),
        "peak_before": max(peak_before.values()) if peak_before else 0.0,
        "peak_after": max(peak_after.values()) if peak_after else 0.0,
        "moved_cust": int(moved_m_full.sum()),
        "moved_pct": float(moved_m_full.sum()) / max(1, len(opt)) * 100.0,
        "moved_vol": float(opt.loc[moved_m_full, cfg.vol_col].sum()),
        "std_before": float(np.std(list(peak_before.values()))) if peak_before else 0.0,
        "std_after": float(np.std(list(peak_after.values()))) if peak_after else 0.0,
    }

    return ZoningResult(
        result_df=opt, stops_df=assigned_stops, daily_matrix=daily_matrix,
        daily_stops=daily_stops_matrix, final_daily=final_daily,
        targets=targets, loads=loads, core_ratio_used=float(ratio_used),
        metrics=metrics, warnings=warnings, infos=infos
    )

# =====================================================================================
#  SECTION 9 — DATA IMPORT & MAPPING UI (ENTERPRISE SIDEBAR)
# =====================================================================================
st.title("🚛 Smart Route Rebalancer — Enterprise")
st.markdown("<div style='color:#64748B; font-weight:600; margin-bottom:1rem;'>ระบบวิเคราะห์และตัดสายส่งน้ำอัตโนมัติ (Multi-Depot Hub-and-Spoke Architecture)</div>", unsafe_allow_html=True)

st.sidebar.markdown("### 📁 1. นำเข้าข้อมูลบริษัท")
sheet_url = st.sidebar.text_input("🔗 ลิงก์ Google Sheets:", value="https://docs.google.com/spreadsheets/d/1PqliJwDoK27SCAj2nlcwkeHmDOdKEQqVcpUaRnQaTWc/edit?usp=sharing", placeholder="วางลิงก์ที่นี่...")
raw_gid = st.sidebar.text_input("แท็บชีต (GID):", value="0")
gm = re.search(r'gid=([0-9]+)', raw_gid)
sheet_gid = gm.group(1) if gm else ("".join(filter(str.isdigit, raw_gid)) or "0")

@st.cache_data(ttl=300, show_spinner=False)
def load_sheet(url: str, gid: str):
    try:
        m = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
        if not m:
            return None, "ลิงก์ Google Sheets ไม่ถูกต้อง"
        u = f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv&gid={gid}"
        d = pd.read_csv(u, dtype=str)
        if d.empty:
            return None, "ไม่พบข้อมูลในแท็บนี้"
        return d, None
    except Exception as e:
        return None, f"เกิดข้อผิดพลาด: {e}"

df = None
if sheet_url:
    key = f"{sheet_url}::{sheet_gid}"
    if st.session_state.get('raw_key') != key:
        ph = st.empty()
        show_loader(ph, "กำลังเชื่อมต่อฐานข้อมูลองค์กร... 💧")
        raw, err = load_sheet(sheet_url, sheet_gid)
        st.session_state.update({'raw_df': raw, 'raw_err': err, 'raw_key': key})
        reset_results()
        ph.empty()
    df = st.session_state.get('raw_df')
    if df is None and st.session_state.get('raw_err'):
        st.sidebar.error(f"❌ {st.session_state['raw_err']}")

if df is None or df.empty:
    st.info("👈 กรุณาวางลิงก์ Google Sheets ที่แถบเมนูด้านซ้าย เพื่อเริ่มต้นใช้งาน")
    st.stop()

df = df.copy()
cols = df.columns.tolist()

# =====================================================================================
#  STEP 2: SELECT DEPOT / BRANCH FIRST (RIGHT AFTER LOADING SHEET)
# =====================================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏢 2. เลือกคลัง/สาขา (Depot Hub)")

depot_col_guess = guess_col(['คลัง', 'สาขา', 'depot'], cols, allow_none=True)
selected_depot = "ทั้งหมด (All Depots)"

if depot_col_guess and depot_col_guess in df.columns:
    depot_list = ["ทั้งหมด (All Depots)"] + sorted(df[depot_col_guess].dropna().astype(str).unique().tolist())
    selected_depot = st.sidebar.selectbox("เลือกสาขาที่ต้องการทำสายส่ง:", depot_list, on_change=reset_results)
    if selected_depot != "ทั้งหมด (All Depots)":
        df = df[df[depot_col_guess].astype(str).str.strip() == selected_depot].reset_index(drop=True)

# =====================================================================================
#  STEP 3: COLUMN MAPPING UI
# =====================================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 3. ยืนยันคอลัมน์ข้อมูลจากชีต")
def sel(label, keys, allow_none=False):
    g = guess_col(keys, cols, allow_none=allow_none)
    if allow_none:
        opts = ["-- ไม่มี --"] + cols
        i = (cols.index(g) + 1) if g in cols else 0
        v = st.sidebar.selectbox(label, opts, index=i, on_change=reset_results)
        return None if v == "-- ไม่มี --" else v
    i = cols.index(g) if g in cols else 0
    return st.sidebar.selectbox(label, cols, index=i, on_change=reset_results)

gps_col = sel("คอลัมน์พิกัด GPS:", ['พิกัด', 'gps', 'lat'])
depot_col = sel("คอลัมน์คลัง/สาขา:", ['คลัง', 'สาขา', 'depot'], allow_none=True)
vol_col = sel("คอลัมน์ยอดส่งเฉลี่ย (ถัง/เดือน):", ['ยอดเฉลี่ย', 'ยอดส่ง', 'volume'])
truck_col = sel("คอลัมน์เบอร์รถ:", ['เบอร์รถ', 'รถ', 'truck'])
day_col = sel("📅 คอลัมน์รอบส่ง/วัน:", ['รอบส่ง', 'วัน', 'day'])
id_col = sel("คอลัมน์รหัสสมาชิก:", ['รหัสสมาชิก', 'รหัส', 'id'])
vip_col = sel("คอลัมน์สถานะ (VIP/ปกติ):", ['สถานะ', 'vip'], allow_none=True)
name_col = sel("คอลัมน์ชื่อลูกค้า:", ['ชื่อ', 'name'], allow_none=True)

# Auto-extract Lat, Lon from single GPS column
def split_lat_lon(val):
    try:
        parts = str(val).split(',')
        if len(parts) >= 2:
            return float(parts[0].strip()), float(parts[1].strip())
    except:
        pass
    return np.nan, np.nan

coords_extracted = df[gps_col].apply(split_lat_lon)
df["_lat_clean"] = [c[0] for c in coords_extracted]
df["_lon_clean"] = [c[1] for c in coords_extracted]

# ใช้งานค่ายอดส่งเฉลี่ยจากคอลัมน์ที่เลือกทันทีโดยไม่ต้องคูณเบิ้ล
df[vol_col] = pd.to_numeric(df[vol_col], errors='coerce').fillna(0).astype(float)
df[truck_col] = df[truck_col].astype(str).str.strip()
df[id_col] = df[id_col].astype(str).str.strip()
df['VIP_Status'] = df[vip_col].astype(str).str.strip() if vip_col else 'ปกติ'

# Set Depot Lat/Lon Hub dynamically based on filtered branch data
depot_lat, depot_lon = float(df["_lat_clean"].mean()), float(df["_lon_clean"].mean())
depot_sub = df.dropna(subset=["_lat_clean", "_lon_clean"])
if not depot_sub.empty:
    depot_lat, depot_lon = float(depot_sub["_lat_clean"].mean()), float(depot_sub["_lon_clean"].mean())

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 พิกัดศูนย์กลางคลัง (Depot Hub)")
depot_lat = st.sidebar.number_input("ละติจูดคลัง (Depot Lat):", value=depot_lat, format="%6.6f", on_change=reset_results)
depot_lon = st.sidebar.number_input("ลองจิจูดคลัง (Depot Lon):", value=depot_lon, format="%6.6f", on_change=reset_results)

df = df.dropna(subset=["_lat_clean", "_lon_clean"]).reset_index(drop=True)
df = df[~df[truck_col].str.lower().isin(NO_TRUCK_TOKENS)].reset_index(drop=True)
available_trucks = clean_truck_ids(df[truck_col].unique())

st.sidebar.markdown("---")
st.sidebar.markdown("### 📏 4. เพดานควบคุมของบริษัท")
monthly_cap = st.sidebar.number_input("ความจุอ้างอิง (ถัง/เดือน/คัน) = 100%", 1000.0, 30000.0, DEFAULT_MONTHLY_CAPACITY, 40.0, on_change=reset_results)
daily_cap = st.sidebar.number_input("เพดานยอดส่งต่อวัน (ถัง/วัน)", 50.0, 500.0, float(DEFAULT_DAILY_CONTROL_CAP), 1.0, on_change=reset_results)
cap_units = daily_cap * DAYS_PER_MONTH

base_cfg = ZoningConfig(lat_col="_lat_clean", lon_col="_lon_clean", vol_col=vol_col, truck_col=truck_col,
                        id_col=id_col, day_col=day_col, name_col=name_col,
                        monthly_capacity=monthly_cap, daily_control_cap=daily_cap,
                        depot_lat=depot_lat, depot_lon=depot_lon)

diag = diagnose_fleet(df, base_cfg)
overloaded = diag.loc[diag['สถานะ'] == '🔴 เกินเพดาน', 'เบอร์รถ'].tolist() if not diag.empty else []

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚫 5. ล็อกรถที่ไม่ต้องคำนวณ")
exclude_trucks = st.sidebar.multiselect("เลือกรถที่ต้องการล็อกไว้ (ห้ามแตะต้อง):", options=available_trucks, default=[], on_change=reset_results)
processable_trucks = [t for t in available_trucks if t not in exclude_trucks]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚚 6. เลือกรถต้นทาง (Donors)")
dissolve_trucks = st.sidebar.multiselect("🗑️ ยุบทั้งคัน", options=processable_trucks, default=[], on_change=reset_results)
relieve_default = [t for t in overloaded if t in processable_trucks and t not in dissolve_trucks]
relieve_trucks = st.sidebar.multiselect("✂️ ดึงงานออกบางส่วน", options=[t for t in processable_trucks if t not in dissolve_trucks], default=relieve_default, on_change=reset_results)

new_trucks_raw = st.sidebar.text_input("➕ เบอร์รถคันใหม่", value="", placeholder="เช่น 15112, 15113", on_change=reset_results)
new_trucks = [t.strip() for t in new_trucks_raw.split(',') if t.strip() and t.strip() not in available_trucks]

vol_by_truck_all = df.groupby(truck_col)[vol_col].sum().to_dict()
kept_trucks = [t for t in processable_trucks if t not in dissolve_trucks]
active_trucks = kept_trucks + new_trucks

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ 7. เป้าหมายรายคัน (%)")
fingerprint = "|".join([sheet_url, sheet_gid, selected_depot, vol_col, truck_col, day_col, str(sorted(dissolve_trucks)), str(sorted(relieve_trucks)), str(new_trucks), str(sorted(exclude_trucks)), f"{monthly_cap}", f"{daily_cap}", f"{depot_lat}", f"{depot_lon}"])
if st.sidebar.button("🔄 คำนวณเป้าหมายอัตโนมัติ"):
    st.session_state.pop('slider_fp', None)
    reset_results()

if st.session_state.get('slider_fp') != fingerprint:
    for k in [k for k in list(st.session_state.keys()) if k.startswith('slider_') or k.startswith('lock_')]:
        del st.session_state[k]
    tgt_units, _ = compute_default_targets(vol_by_truck_all, dissolve_trucks, kept_trucks, new_trucks, cap_units)
    st.session_state.truck_pcts = {t: float(round(max(0.0, min(200.0, tgt_units.get(t, 0.0) / monthly_cap * 100)), 1)) for t in active_trucks}
    for t in active_trucks:
        st.session_state[f"slider_{t}"] = st.session_state.truck_pcts[t]
    st.session_state['slider_fp'] = fingerprint

def on_slider_change(changed: str):
    new_val = max(0.0, min(200.0, st.session_state.get(f"slider_{changed}", 0.0)))
    old = st.session_state.truck_pcts.get(changed, new_val)
    diff = new_val - old
    if abs(diff) < 0.01:
        return
    free = [t for t in active_trucks if t != changed and not st.session_state.get(f"lock_{t}", False)]
    if free:
        share = diff / len(free)
        for t in free:
            v = round(max(0.0, min(200.0, st.session_state.truck_pcts.get(t, 0.0) - share)), 1)
            st.session_state.truck_pcts[t] = v
            st.session_state[f"slider_{t}"] = v
    st.session_state.truck_pcts[changed] = round(new_val, 1)
    reset_results()

target_pcts: Dict[str, float] = {}
for t in active_trucks:
    c1, c2 = st.sidebar.columns([3, 1.2])
    with c2:
        st.checkbox("🔒", key=f"lock_{t}", on_change=reset_results)
    with c1:
        v = st.slider(f"{'🆕 ' if t in new_trucks else ''}รถ {t} (%)", 0.0, 200.0, step=0.1, key=f"slider_{t}", on_change=on_slider_change, args=(t,))
        target_pcts[t] = v
        st.session_state.truck_pcts[t] = v

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔒 8. ล็อก Key Account & กฎ")
manual_vips = st.sidebar.multiselect("รหัสสมาชิกห้ามย้ายสาย", options=df[id_col].unique().tolist(), default=[], on_change=reset_results)
core_ratio_pct = st.sidebar.slider("สัดส่วนแกนกลาง (Core %)", 0, 100, 65, 5, on_change=reset_results)
tol_pct = st.sidebar.number_input("ค่าเผื่อเป้าหมาย (%)", 0.0, 50.0, 5.0, 0.5, on_change=reset_results)

cfg = ZoningConfig(lat_col="_lat_clean", lon_col="_lon_clean", vol_col=vol_col, truck_col=truck_col,
                    id_col=id_col, day_col=day_col, name_col=name_col,
                    monthly_capacity=monthly_cap, daily_control_cap=daily_cap,
                    depot_lat=depot_lat, depot_lon=depot_lon, core_ratio=float(core_ratio_pct),
                    tol_pct=float(tol_pct))

if st.sidebar.button("🚀 ประมวลผลจัดสายส่งใหม่", use_container_width=True):
    ph = st.empty()
    show_loader(ph, "กำลังประมวลผลจัดกลุ่มเส้นทางอัจฉริยะ (Multi-Depot Hub)... 💧")
    res = run_multi_donor_zoning(df, cfg, target_pcts, dissolve_trucks, relieve_trucks, new_trucks, manual_vips, exclude_trucks)
    st.session_state['result'] = res
    st.session_state['zoning_cfg_used'] = cfg
    ph.empty()

# =====================================================================================
#  SECTION 10 — MODERN ENTERPRISE TABS UI (ORGANIZED DASHBOARD)
# =====================================================================================
tab1, tab2, tab3 = st.tabs([
    "📊 1. ภาพรวมและวินิจฉัย (Dashboard)", 
    "🗺️ 2. แผนที่เปรียบเทียบ (Before vs. After)", 
    "📋 3. ตารางโหลดรายวันและรายละเอียด (Reports)"
])

with tab1:
    st.markdown(f"### 📈 ตัวชี้วัดผลลัพธ์เพื่อการตัดสินใจ ({selected_depot})")
    if 'result' in st.session_state:
        res: ZoningResult = st.session_state['result']
        over_after = int(res.metrics['over_after'])
        over_diff = int(res.metrics['over_after'] - res.metrics['over_before'])
        peak_after = res.metrics['peak_after']
        peak_diff = res.metrics['peak_after'] - res.metrics['peak_before']
        compact_diff = res.metrics['compact_after_km'] - res.metrics['compact_before_km']

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_enterprise_metric("รถเกินเพดาน", f"{over_after} คัน", delta=f"{over_diff:+d}", status="danger" if over_after > 0 else "good")
        with k2:
            render_enterprise_metric("โหลดสูงสุด", f"{peak_after:,.0f} ถัง/วัน", delta=f"{peak_diff:+,.0f} ถัง", status="warning" if peak_after > daily_cap else "good")
        with k3:
            render_enterprise_metric("ระยะทางวิ่งเฉลี่ยจากคลัง", f"{res.metrics['compact_after_km']:.2f} กม.", delta=f"{compact_diff:+.2f} กม.", status="good" if compact_diff <= 0 else "warning")
        with k4:
            render_enterprise_metric("ลูกค้าที่โยกย้าย", f"{int(res.metrics['moved_cust']):,} ราย", delta=f"{res.metrics['moved_pct']:.1f}% ของคลัง", status="normal")
    else:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_enterprise_metric("รถทั้งหมด", f"{len(diag)} คัน", status="normal")
        with k2:
            render_enterprise_metric("รถเกินเพดาน", f"{len(overloaded)} คัน", delta=f"เพดาน {daily_cap:,.0f} ถัง/วัน", status="danger" if overloaded else "good")
        with k3:
            render_enterprise_metric("ยอดรวมทั้งคลัง", f"{df[vol_col].sum():,.0f} ถัง/เดือน", status="normal")
        with k4:
            excess_total = float(diag['ส่วนเกิน/วัน'].sum()) * DAYS_PER_MONTH if not diag.empty else 0.0
            render_enterprise_metric("ยอดส่วนเกินที่ต้องย้าย", f"{excess_total:,.0f} ถัง/เดือน", status="warning" if excess_total > 0 else "good")

    st.markdown("---")
    st.markdown("### 🩺 ผลวินิจฉัยสถานะรถปัจจุบัน (ก่อนปรับ)")
    diag_html = '<div class="custom-table-container"><table class="custom-table"><thead><tr><th>เบอร์รถ</th><th>จำนวนลูกค้า</th><th>ยอด/เดือน</th><th>ภาระงาน(%)</th><th>โหลดสูงสุด/วัน</th><th>จุดจอด</th><th>ส่วนเกิน/วัน</th><th>สถานะ</th></tr></thead><tbody>'
    for _, row in diag.iterrows():
        is_over = "🔴 เกินเพดาน" in str(row['สถานะ'])
        is_warn = "🟡 ควรเลี่ยง" in str(row['สถานะ'])
        vol_cls = "text-danger" if is_over else ("text-primary" if not is_warn else "")
        status_color = "#DC2626" if is_over else ("#D97706" if is_warn else "#16A34A")
        diag_html += f'<tr><td style="font-weight:700;">{row["เบอร์รถ"]}</td><td>{row["จำนวนลูกค้า"]:,}</td><td>{row["ยอด/เดือน"]:,}</td><td>{row["ภาระงาน(%)"]}%</td><td class="{vol_cls}">{row["โหลดสูงสุด/วัน"]:,}</td><td>{row["จุดจอด"]:,}</td><td style="color:{"#DC2626" if row["ส่วนเกิน/วัน"] > 0 else "#475569"}; font-weight:700;">{row["ส่วนเกิน/วัน"]:,}</td><td><span style="color:{status_color}; font-weight:700;">{row["สถานะ"]}</span></td></tr>'
    diag_html += "</tbody></table></div>"
    st.markdown(diag_html, unsafe_allow_html=True)

with tab2:
    st.markdown(f"### 🗺️ แผนที่เปรียบเทียบเชิงพื้นที่ ({selected_depot})")
    if 'result' in st.session_state:
        res: ZoningResult = st.session_state['result']
        rdf = res.result_df
        distinct_colors = ['#2563EB', '#16A34A', '#F59E0B', '#9333EA', '#0284C7', '#14B8A6', '#6366F1', '#84CC16', '#EC4899', '#06B6D4']
        all_truck_keys = sorted(list(set(clean_truck_ids(df[truck_col].unique()) + active_trucks + exclude_trucks)))
        color_map = {t: ('#DC2626' if t in new_trucks else distinct_colors[i % len(distinct_colors)]) for i, t in enumerate(all_truck_keys)}
        color_map[OVERFLOW_LABEL] = '#64748B'

        sel_t_view = st.multiselect("🔍 กรองดูเฉพาะเบอร์รถ (เว้นว่างเพื่อดูทั้งหมด):", options=all_truck_keys, default=[])
        df_b = df if not sel_t_view else df[df[truck_col].isin(sel_t_view)]
        df_a = rdf[rdf['เบอร์รถใหม่'] != OVERFLOW_LABEL] if not sel_t_view else rdf[rdf['เบอร์รถใหม่'].isin(sel_t_view)]

        center_lat, center_lon = (depot_lat, depot_lon) if depot_lat != 0 else (float(rdf["_lat_clean"].mean()), float(rdf["_lon_clean"].mean()))
        mcol1, mcol2 = st.columns(2)

        def make_popup(r, is_after=False):
            c_id = str(r[id_col]) if id_col in r else "-"
            c_name = "-"
            if name_col and name_col in r.index and pd.notna(r[name_col]):
                c_name = str(r[name_col])
            vol_m = float(pd.to_numeric(r[vol_col], errors="coerce") or 0.0) if vol_col in r else 0.0
            orig_t = str(r[truck_col]).strip() if truck_col in r else "-"
            new_t = str(r["เบอร์รถใหม่"]).strip() if "เบอร์รถใหม่" in r else orig_t
            return f"<b>รหัสสมาชิก:</b> {c_id}<br><b>ชื่อ:</b> {c_name}<br><b>ยอด:</b> {vol_m:,.0f} ถัง<br><b>สาย:</b> {orig_t if not is_after else orig_t + ' → ' + new_t}"

        with mcol1:
            st.markdown("<div style='text-align:center; font-weight:700; color:#024D7B; margin-bottom:6px;'>📍 ก่อนปรับ (Before)</div>", unsafe_allow_html=True)
            mb = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")
            if depot_lat != 0 and depot_lon != 0:
                folium.Marker([depot_lat, depot_lon], popup=f"<b>📌 {selected_depot}</b>", icon=folium.Icon(color='red', icon='home', prefix='fa')).add_to(mb)
            for _, r in df_b.iterrows():
                folium.CircleMarker([r["_lat_clean"], r["_lon_clean"]], radius=5, color=color_map.get(str(r[truck_col]), "#64748B"), fill=True, fillColor=color_map.get(str(r[truck_col]), "#64748B"), fill_opacity=0.8, popup=folium.Popup(make_popup(r, False), max_width=250)).add_to(mb)
            components.html(mb.get_root().render(), height=650)

        with mcol2:
            st.markdown("<div style='text-align:center; font-weight:700; color:#024D7B; margin-bottom:6px;'>✨ หลังปรับ (Hub Routing)</div>", unsafe_allow_html=True)
            ma = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")
            if depot_lat != 0 and depot_lon != 0:
                folium.Marker([depot_lat, depot_lon], popup=f"<b>📌 {selected_depot}</b>", icon=folium.Icon(color='red', icon='home', prefix='fa')).add_to(ma)
            for _, r in df_a.iterrows():
                folium.CircleMarker([r["_lat_clean"], r["_lon_clean"]], radius=5, color=color_map.get(str(r["เบอร์รถใหม่"]), "#64748B"), fill=True, fillColor=color_map.get(str(r["เบอร์รถใหม่"]), "#64748B"), fill_opacity=0.8, popup=folium.Popup(make_popup(r, True), max_width=250)).add_to(ma)
            components.html(ma.get_root().render(), height=650)
    else:
        st.info("💡 กรุณากดปุ่ม '🚀 ประมวลผลจัดสายส่งใหม่' ที่เมนูด้านซ้ายเพื่อแสดงแผนที่")

with tab3:
    st.markdown("### 📅 ตารางวิเคราะห์โหลดรายวัน & รายรายละเอียดการโยกย้าย")
    if 'result' in st.session_state:
        res: ZoningResult = st.session_state['result']
        rdf = res.result_df

        def format_cell(val, cap):
            rv = int(round(val))
            return f'<td class="text-danger">{rv:,}</td>' if rv > cap else f'<td class="text-primary">{rv:,}</td>'

        d_html = '<div class="custom-table-container"><table class="custom-table"><thead><tr><th>เบอร์รถ</th><th>จันทร์</th><th>อังคาร</th><th>พุธ</th><th>พฤหัสฯ</th><th>ศุกร์</th><th>เสาร์</th><th>โหลดสูงสุด</th><th>สถานะ</th></tr></thead><tbody>'
        for t in sorted(list(set(active_trucks + exclude_trucks))):
            dv = res.final_daily.get(t, np.zeros(WORKING_DAYS))
            pk = int(round(dv.max()))
            ov = pk > cfg.daily_control_cap
            badge = '<span style="color:#DC2626; font-weight:700;">🔴 เกินเกณฑ์</span>' if ov else '<span style="color:#16A34A; font-weight:700;">🟢 ผ่านเกณฑ์</span>'
            d_html += f'<tr><td style="font-weight:700;">{t}</td>{format_cell(dv[0], cfg.daily_control_cap)}{format_cell(dv[1], cfg.daily_control_cap)}{format_cell(dv[2], cfg.daily_control_cap)}{format_cell(dv[3], cfg.daily_control_cap)}{format_cell(dv[4], cfg.daily_control_cap)}{format_cell(dv[5], cfg.daily_control_cap)}<td style="font-weight:700;">{pk:,}</td><td>{badge}</td></tr>'
        d_html += "</tbody></table></div>"
        st.markdown(d_html, unsafe_allow_html=True)

        st.markdown("### 📋 รายละเอียดการโยกย้ายลูกค้าทั้งหมด")
        f_cols = [id_col]
        if name_col and name_col in rdf.columns:
            f_cols.append(name_col)
        f_cols.extend([vol_col, truck_col, "เบอร์รถใหม่", day_col, "วันจัดส่ง(ใหม่)", "สถานะการย้ายวัน", "สถานะ"])
        st.dataframe(rdf[f_cols].rename(columns={truck_col: "เบอร์รถเดิม"}), use_container_width=True)

        csv_data = rdf[f_cols].to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 ดาวน์โหลดผลการจัดสายส่ง (CSV)", csv_data, 'sprinkle_rebalance_enterprise.csv', 'text/csv', use_container_width=True)
    else:
        st.info("💡 กรุณากดปุ่ม '🚀 ประมวลผลจัดสายส่งใหม่' เพื่อดูตารางรายงานและดาวน์โหลดข้อมูล")
