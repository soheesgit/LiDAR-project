# frame_processing.py
# 프레임 1장 읽어 ROI 필터링, 정적 occupancy 생성, 차량 클러스터링/트래킹 반영, delta 맵 생성을 담당하는 프레임 처리 모듈

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
from pathlib import Path
import numpy as np

from tracking import TrackManager
from kitti_io import read_bin_xyzr, read_sem_labels

@dataclass
class RoiFrame:
    x: np.ndarray
    y: np.ndarray
    sem: np.ndarray


@dataclass
class SequenceState:
    unique_sets: list[list[set[int]]]
    dwell: np.ndarray
    sum_v: np.ndarray
    sum_v2: np.ndarray
    cnt_v: np.ndarray
    static_dwell: np.ndarray
    static_change_count: np.ndarray
    static_prev_occ: np.ndarray | None = None


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


@dataclass
class FrameProcessStepResult:
    roi_count: int
    occ: np.ndarray
    dwell_delta: np.ndarray
    sum_v_delta: np.ndarray
    sum_v2_delta: np.ndarray
    cnt_v_delta: np.ndarray
    obs: list


def process_frame_step(
    *,
    t: int,
    bin_path: Path,
    lbl_path: Path,
    state: SequenceState,
    tracker,
    H: int,
    W: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    res: float,
    vehicle_ids: set,
    static_ids: set,
    eps: float,
    min_samples: int,
    min_pts: int,
    chk_first: int,
    chk_every: int,
    xy_to_cell_fn,
    cluster_vehicle_xy_fn,
) -> Optional[FrameProcessStepResult]:
    pts = read_bin_xyzr(bin_path)
    sem = read_sem_labels(lbl_path)

    if pts.shape[0] != sem.shape[0]:
        print(f"  [SKIP] size mismatch at {bin_path.stem}")
        return None

    roi_frame = apply_roi_filter(
        pts=pts,
        sem=sem,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )
    x, y, sem = roi_frame.x, roi_frame.y, roi_frame.sem

    veh_m = np.isin(sem, list(vehicle_ids))
    static_m = np.isin(sem, list(static_ids))

    do_chk = (t < chk_first) or (chk_every > 0 and (t % chk_every == 0))
    if do_chk:
        print(
            f"[CHK t={t}] roi_pts={len(x)} veh_pts={int(veh_m.sum())} static_pts={int(static_m.sum())}"
        )

    occ = build_static_occupancy(
        x=x,
        y=y,
        sem=sem,
        static_ids=static_ids,
        H=H,
        W=W,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        res=res,
        xy_to_cell_fn=xy_to_cell_fn,
    )
    apply_static_occupancy(state, occ)

    veh_result = build_vehicle_deltas(
        x=x,
        y=y,
        sem=sem,
        vehicle_ids=vehicle_ids,
        H=H,
        W=W,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        res=res,
        tracker=tracker,
        cluster_vehicle_xy_fn=cluster_vehicle_xy_fn,
        xy_to_cell_fn=xy_to_cell_fn,
        eps=eps,
        min_samples=min_samples,
        min_pts=min_pts,
    )
    apply_vehicle_result(state, veh_result)

    dwell_delta = veh_result.dwell_delta
    sum_v_delta = veh_result.sum_v_delta
    sum_v2_delta = veh_result.sum_v2_delta
    cnt_v_delta = veh_result.cnt_v_delta
    obs = veh_result.obs
    roi_count = veh_result.roi_count
    seen_tid = veh_result.seen_tid

    if do_chk:
        print(
            f"[CHK t={t}] clusters={veh_result.cluster_count} "
            f"centers={len(veh_result.centers)} "
            f"(eps={eps}, min_samples={min_samples}, min_pts={min_pts})"
        )
        print(
            f"[CHK t={t}] tracks_alive={len(tracker.tracks)} "
            f"updated={veh_result.updated_track_count}"
        )

        nz = int(np.count_nonzero(dwell_delta))
        nz_cntv = int(np.count_nonzero(cnt_v_delta))
        print(
            f"[CHK t={t}] roi_count={roi_count} seen_tid={len(seen_tid)} "
            f"dwell_delta_nz={nz} cnt_v_delta_nz={nz_cntv}"
        )

    m_cnt = (cnt_v_delta > 0)
    n_cnt = int(np.sum(m_cnt))
    n_sum_finite = int(np.sum(np.isfinite(sum_v_delta[m_cnt]))) if n_cnt > 0 else 0
    n_sum_zero = int(np.sum(sum_v_delta[m_cnt] == 0.0)) if n_cnt > 0 else 0
    n_sum_bad = int(np.sum(~np.isfinite(sum_v_delta[m_cnt]))) if n_cnt > 0 else 0

    if n_cnt > 0 and np.isfinite(sum_v_delta[m_cnt]).any():
        approx_vals = sum_v_delta[m_cnt & (cnt_v_delta == 1)]
        approx_vals = approx_vals[np.isfinite(approx_vals)]
        if approx_vals.size > 0:
            v_lo = float(np.min(approx_vals))
            v_hi = float(np.max(approx_vals))
        else:
            v_lo = v_hi = float("nan")
    else:
        v_lo = v_hi = float("nan")

    print(
        f"[DBG][t={t}] delta cnt_cells={n_cnt}, sum_finite={n_sum_finite}, "
        f"sum_zero={n_sum_zero}, sum_bad={n_sum_bad}, approx_v_range=[{v_lo:.3f},{v_hi:.3f}]"
    )

    return FrameProcessStepResult(
        roi_count=roi_count,
        occ=occ,
        dwell_delta=dwell_delta,
        sum_v_delta=sum_v_delta,
        sum_v2_delta=sum_v2_delta,
        cnt_v_delta=cnt_v_delta,
        obs=obs,
    )


def init_sequence_state(H: int, W: int) -> SequenceState:
    return SequenceState(
        unique_sets=[[set() for _ in range(W)] for _ in range(H)],
        dwell=np.zeros((H, W), dtype=np.int32),
        sum_v=np.zeros((H, W), dtype=np.float32),
        sum_v2=np.zeros((H, W), dtype=np.float32),
        cnt_v=np.zeros((H, W), dtype=np.int32),
        static_dwell=np.zeros((H, W), dtype=np.int32),
        static_change_count=np.zeros((H, W), dtype=np.int32),
        static_prev_occ=None,
    )


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


def apply_static_occupancy(state: SequenceState, occ: np.ndarray) -> None:
    state.static_dwell += occ.astype(np.int32)

    if state.static_prev_occ is not None:
        diff = np.logical_xor(occ, state.static_prev_occ)
        state.static_change_count += diff.astype(np.int32)

    state.static_prev_occ = occ


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

        if tr.has_velocity:
            v = float(tr.speed)
            if np.isfinite(v) and v >= 0:
                sum_v_delta[iyy, ixx] += v
                sum_v2_delta[iyy, ixx] += v * v
                cnt_v_delta[iyy, ixx] += 1

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


def apply_vehicle_result(state: SequenceState, result: VehicleDeltaResult) -> None:
    state.dwell += result.dwell_delta
    state.sum_v += result.sum_v_delta
    state.sum_v2 += result.sum_v2_delta
    state.cnt_v += result.cnt_v_delta

    for tid_int, iyy, ixx, _spd in result.obs:
        state.unique_sets[iyy][ixx].add(int(tid_int))
