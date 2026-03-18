from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple
import numpy as np

from src.tracking import TrackManager


@dataclass
class RoiFrame:
    x: np.ndarray
    y: np.ndarray
    sem: np.ndarray


@dataclass
class VehicleDeltaResult:
    dwell_delta: np.ndarray
    sum_v_delta: np.ndarray
    sum_v2_delta: np.ndarray
    cnt_v_delta: np.ndarray
    obs: list[tuple[int, int, int, float]]
    roi_count: int
    seen_tid: set[int]
    centers: list[tuple[float, float]]
    sizes: list[int]
    cluster_count: int
    updated_track_count: int


def apply_roi_filter(
    pts: np.ndarray,
    sem: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> RoiFrame:
    x, y, z, _ = pts.T
    m_roi = (
        (x >= x_min) & (x < x_max) &
        (y >= y_min) & (y < y_max) &
        (z >= z_min) & (z <= z_max)
    )
    return RoiFrame(
        x=x[m_roi],
        y=y[m_roi],
        sem=sem[m_roi],
    )


def build_static_occupancy(
    x: np.ndarray,
    y: np.ndarray,
    sem: np.ndarray,
    static_ids: Iterable[int],
    H: int,
    W: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    res: float,
    xy_to_cell_fn,
) -> np.ndarray:
    static_m = np.isin(sem, list(static_ids))
    occ = np.zeros((H, W), dtype=bool)

    if not static_m.any():
        return occ

    xs, ys = x[static_m], y[static_m]
    iy_s, ix_s, m_s, _, _ = xy_to_cell_fn(xs, ys, x_min, x_max, y_min, y_max, res)
    if m_s.any():
        occ[iy_s[m_s], ix_s[m_s]] = True
    return occ


def build_vehicle_deltas(
    x: np.ndarray,
    y: np.ndarray,
    sem: np.ndarray,
    vehicle_ids: Iterable[int],
    H: int,
    W: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    res: float,
    tracker: TrackManager,
    cluster_vehicle_xy_fn,
    xy_to_cell_fn,
    eps: float,
    min_samples: int,
    min_pts: int,
    dwell: np.ndarray,
    unique_sets,
    sum_v: np.ndarray,
    sum_v2: np.ndarray,
    cnt_v: np.ndarray,
) -> VehicleDeltaResult:
    dwell_delta = np.zeros((H, W), dtype=np.int32)
    sum_v_delta = np.zeros((H, W), dtype=np.float32)
    sum_v2_delta = np.zeros((H, W), dtype=np.float32)
    cnt_v_delta = np.zeros((H, W), dtype=np.int32)
    obs: list[tuple[int, int, int, float]] = []

    veh_m = np.isin(sem, list(vehicle_ids))
    xy_veh = (
        np.stack([x[veh_m], y[veh_m]], axis=1)
        if veh_m.any()
        else np.zeros((0, 2), dtype=np.float32)
    )

    clusters = cluster_vehicle_xy_fn(xy_veh, eps, min_samples, min_pts)

    centers: list[tuple[float, float]] = []
    sizes: list[int] = []
    for c in clusters:
        cx = float(np.median(c[:, 0]))
        cy = float(np.median(c[:, 1]))
        centers.append((cx, cy))
        sizes.append(int(c.shape[0]))

    tracks = tracker.update(centers, sizes)
    updated_track_count = sum(1 for tr in tracks if tr.just_updated)

    roi_count = 0
    seen_tid: set[int] = set()

    for tr in tracks:
        if not tr.just_updated:
            continue

        xy1 = np.array([[tr.center[0], tr.center[1]]], dtype=np.float32)
        iy, ix, m, _, _ = xy_to_cell_fn(
            xy1[:, 0], xy1[:, 1], x_min, x_max, y_min, y_max, res
        )
        if not m.any():
            continue

        iyy, ixx = int(iy[0]), int(ix[0])

        tid_int = int(tr.tid)
        if tid_int not in seen_tid:
            seen_tid.add(tid_int)
            roi_count += 1

        spd = float(tr.speed) if tr.has_velocity else np.nan
        obs.append((tid_int, iyy, ixx, spd))

        dwell_delta[iyy, ixx] += 1
        dwell[iyy, ixx] += 1
        unique_sets[iyy][ixx].add(tid_int)

        if tr.has_velocity:
            v = float(tr.speed)
            if np.isfinite(v) and v >= 0:
                sum_v_delta[iyy, ixx] += v
                sum_v2_delta[iyy, ixx] += v * v
                cnt_v_delta[iyy, ixx] += 1

                sum_v[iyy, ixx] += v
                sum_v2[iyy, ixx] += v * v
                cnt_v[iyy, ixx] += 1

    return VehicleDeltaResult(
        dwell_delta=dwell_delta,
        sum_v_delta=sum_v_delta,
        sum_v2_delta=sum_v2_delta,
        cnt_v_delta=cnt_v_delta,
        obs=obs,
        roi_count=roi_count,
        seen_tid=seen_tid,
        centers=centers,
        sizes=sizes,
        cluster_count=len(clusters),
        updated_track_count=updated_track_count,
    )
