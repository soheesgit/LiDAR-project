from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import numpy as np
import os

from src.encoder_utils import (
    _safe_nanmean,
    _normalize_01,
    _topk_mask,
    _topk_mask_on_values,
    _density_from_sumcount,
    _density_from_meancount,
)


@dataclass
class EncoderConfig:
    # Density (면적당 차량 대수) 정규화 설정
    density_cap_cell: float = 2.0
    cell_size_m: float = 0.8
    n_frames: int = 300

    # ROI 안 차량 대수 기반 density 정규화(1.0이 되는 기준 차량 수)
    roi_cap_vehicles: float = 20.0

    # speed 유효성 게이트
    speed_reliable_ratio_min: float = 0.30

    # 속도 기준 (m/s)
    v_low: float = 2.0      # 정체로 보일 만큼 느림
    v_ok: float = 6.0       # 정상 흐름으로 보일 만큼 빠름
    v_stop: float = 1.5     # 저속/정지 판정 기준 (m/s)
    tol_cell: int = 1       # 주변 허용 셀 반경 (1이면 8-neighborhood)

    # Occupancy(체류) 기준 (0~1 정규화 후)
    occ_high: float = 0.6

    stopped_ratio_high: float = 0.5   # 윈도우 관측(속도 있는) 차량 중 저속/정차 비율이 이 이상이면 '정체' 강화
    speed_std_mix_high: float = 2.0

    # Focus
    topk_ratio: Optional[float] = None
    topk_ratio_dyn: float = 0.05
    topk_ratio_stat: float = 0.05
    topk_ratio_speed: float = 0.10

    # Static / ego / occlusion
    ego_motion_low: float = 0.10
    scr_low: float = 0.10
    sd_high: float = 0.60

    # density 기준
    dens_low: float = 0.20
    dens_cong: float = 0.60
    min_valid_density: float = 1e-6

    # speed 샘플 기반 유효성
    min_speed_samples: int = 5              # cnt_v >= 5인 셀만 속도 신뢰
    require_speed_samples: bool = True      # speed_samples_map 들어오면 샘플로 gate

    # debug
    debug: bool = False
    debug_print: bool = True
    debug_speed_nan: bool = True

    # empty 판정
    empty_roi_mean_thr: float = 0.5     # 윈도우 평균 ROI 차량수 < 이 값이면 "거의 없음"
    empty_dens_mean_thr: float = 0.05   # dens_mean < 이 값이면 "거의 없음"

    def __post_init__(self):
        if self.topk_ratio is not None:
            r = float(self.topk_ratio)
            self.topk_ratio_dyn = r
            self.topk_ratio_stat = r


# 디버그 프린트 헬퍼
def _dbg_print(enabled: bool, msg: str):
    if enabled:
        print(msg)


# 안전한 통계 요약: 배열의 min/max/mean 같은 것들을 한 번에 보기 위함
def _dbg_stats(name: str, arr: Optional[np.ndarray], mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if arr is None:
        out[f"{name}_mean"] = float("nan")
        out[f"{name}_min"] = float("nan")
        out[f"{name}_max"] = float("nan")
        out[f"{name}_n"] = 0.0
        return out

    a = arr.astype(np.float32)
    if mask is not None:
        try:
            a = a[mask]
        except Exception:
            a = arr.astype(np.float32)

    finite = np.isfinite(a)
    if not finite.any():
        out[f"{name}_mean"] = float("nan")
        out[f"{name}_min"] = float("nan")
        out[f"{name}_max"] = float("nan")
        out[f"{name}_n"] = float(a.size)
        return out

    out[f"{name}_mean"] = float(np.nanmean(a[finite]))
    out[f"{name}_min"] = float(np.nanmin(a[finite]))
    out[f"{name}_max"] = float(np.nanmax(a[finite]))
    out[f"{name}_n"] = float(a.size)
    return out


# speed_mean이 NaN으로 떨어지는 "원인"을 더 직접적으로 잡아내는 진단 함수
def _dbg_speed_nan_diagnose(
    *,
    mean_speed_map: Optional[np.ndarray],
    std_speed_map: Optional[np.ndarray],
    speed_samples_map: Optional[np.ndarray],
    focus_dyn: np.ndarray,
    reliable_cells: Optional[np.ndarray],
    cfg: EncoderConfig,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # 0) focus 통계
    focus_n = int(np.sum(focus_dyn))
    out["dbg_sp_focus_n"] = float(focus_n)

    # 1) speed_samples_map(=cnt_v) 쪽 진단
    if speed_samples_map is None:
        out["dbg_sp_has_samples_map"] = 0.0
    else:
        out["dbg_sp_has_samples_map"] = 1.0
        ss = speed_samples_map.astype(np.float32)
        out.update(_dbg_stats("dbg_sp_ss_focus", ss, focus_dyn))
        rel = focus_dyn & (ss >= cfg.min_speed_samples)
        out["dbg_sp_reliable_n_from_ss"] = float(int(np.sum(rel)))
        out["dbg_sp_min_speed_samples_thr"] = float(cfg.min_speed_samples)

    # 2) mean_speed_map 쪽 진단
    if mean_speed_map is None:
        out["dbg_sp_has_mean_speed_map"] = 0.0
        return out

    out["dbg_sp_has_mean_speed_map"] = 1.0
    sp = mean_speed_map.astype(np.float32)

    # focus 내 mean_speed finite 비율/개수
    if focus_n > 0:
        sp_focus = sp[focus_dyn]
        finite_focus = np.isfinite(sp_focus)
        out["dbg_sp_focus_finite_n"] = float(int(np.sum(finite_focus)))
        out["dbg_sp_focus_finite_ratio"] = float(np.sum(finite_focus) / max(1, sp_focus.size))
        if finite_focus.any():
            out["dbg_sp_focus_min"] = float(np.nanmin(sp_focus[finite_focus]))
            out["dbg_sp_focus_max"] = float(np.nanmax(sp_focus[finite_focus]))
        else:
            out["dbg_sp_focus_min"] = float("nan")
            out["dbg_sp_focus_max"] = float("nan")
    else:
        out["dbg_sp_focus_finite_n"] = 0.0
        out["dbg_sp_focus_finite_ratio"] = float("nan")
        out["dbg_sp_focus_min"] = float("nan")
        out["dbg_sp_focus_max"] = float("nan")

    # reliable_cells가 있으면(샘플 기반 gate 후) 그 안에서 다시 진단
    if reliable_cells is not None:
        rel_n = int(np.sum(reliable_cells))
        out["dbg_sp_reliable_n"] = float(rel_n)
        if rel_n > 0:
            sp_rel = sp[reliable_cells]
            finite_rel = np.isfinite(sp_rel)
            out["dbg_sp_reliable_finite_n"] = float(int(np.sum(finite_rel)))
            out["dbg_sp_reliable_finite_ratio"] = float(np.sum(finite_rel) / max(1, sp_rel.size))
            if finite_rel.any():
                out["dbg_sp_reliable_min"] = float(np.nanmin(sp_rel[finite_rel]))
                out["dbg_sp_reliable_max"] = float(np.nanmax(sp_rel[finite_rel]))
            else:
                out["dbg_sp_reliable_min"] = float("nan")
                out["dbg_sp_reliable_max"] = float("nan")
        else:
            out["dbg_sp_reliable_finite_n"] = 0.0
            out["dbg_sp_reliable_finite_ratio"] = float("nan")
            out["dbg_sp_reliable_min"] = float("nan")
            out["dbg_sp_reliable_max"] = float("nan")

    # 3) std_speed_map도 같이 보면 "속도는 NaN인데 std는 값이 있다/없다" 같은 힌트가 됨
    if std_speed_map is None:
        out["dbg_sp_has_std_speed_map"] = 0.0
    else:
        out["dbg_sp_has_std_speed_map"] = 1.0
        st = std_speed_map.astype(np.float32)
        # focus 내 std finite 비율
        if focus_n > 0:
            st_focus = st[focus_dyn]
            finite_st = np.isfinite(st_focus)
            out["dbg_std_focus_finite_n"] = float(int(np.sum(finite_st)))
            out["dbg_std_focus_finite_ratio"] = float(np.sum(finite_st) / max(1, st_focus.size))
        else:
            out["dbg_std_focus_finite_n"] = 0.0
            out["dbg_std_focus_finite_ratio"] = float("nan")

    return out


def _compute_density_context(
    *,
    sum_count_map: Optional[np.ndarray],
    mean_count_map: Optional[np.ndarray],
    unique_cnt_map: Optional[np.ndarray],
    dwell_map: Optional[np.ndarray],
    mean_speed_map: Optional[np.ndarray],
    speed_samples_map: Optional[np.ndarray],
    static_change_rate: Optional[np.ndarray],
    static_dwell_map: Optional[np.ndarray],
    roi_count_series: Optional[np.ndarray],
    cfg: EncoderConfig,
    debug: bool,
    dbg_print_on: bool,
    debug_tag: str,
    dbg: Dict[str, float],
) -> Tuple[Dict[str, Any], Optional[Tuple[str, Dict[str, float]]]]:
    ref_shape = None
    for m in (
        dwell_map,
        mean_speed_map,
        speed_samples_map,
        static_change_rate,
        static_dwell_map,
        unique_cnt_map,
        sum_count_map,
        mean_count_map,
    ):
        if m is not None:
            ref_shape = m.shape
            break
    if ref_shape is None:
        ref_shape = (1, 1)  # 최후 fallback

    # 1) Density
    if roi_count_series is not None and roi_count_series.size > 0:
        # 최근 N프레임 ROI 차량 수 평균
        roi_mean = float(np.mean(roi_count_series))
        dens_scalar = np.clip(roi_mean / max(cfg.roi_cap_vehicles, 1e-6), 0.0, 1.0)
        density = np.full(ref_shape, dens_scalar, dtype=np.float32)
        #  디버그: ROI 기반 density 정보
        dbg["dbg_density_mode_roi"] = 1.0
        dbg["dbg_roi_mean"] = float(roi_mean)
        dbg["dbg_dens_scalar"] = float(dens_scalar)
    elif sum_count_map is not None:
        density = _density_from_sumcount(sum_count_map, cfg)
        dbg["dbg_density_mode_sumcount"] = 1.0
    elif mean_count_map is not None:
        density = _density_from_meancount(mean_count_map, cfg)
        dbg["dbg_density_mode_meancount"] = 1.0
    else:
        if unique_cnt_map is not None:
            density = np.clip(
                unique_cnt_map.astype(np.float32) / max(float(cfg.density_cap_cell), 1e-6),
                0.0,
                1.0,
            )
            dbg["dbg_density_mode_unique"] = 1.0
        elif dwell_map is not None:
            density = _normalize_01(dwell_map.astype(np.float32))
            dbg["dbg_density_mode_dwell_fallback"] = 1.0
        else:
            raise ValueError(
                "Need one of {roi_count_series, sum_count_map, mean_count_map, unique_cnt_map, dwell_map}."
            )

    # 2) Occupancy (0~1)
    # - 윈도우/전체 모두에서 얼마나 자주 점유됐나(프레임 비율)로 해석하는 게 안정적
    # - dwell_map이 이미 0~1 비율로 들어오면 그대로 쓰고, dwell_map이 카운트(0~N)로 들어오면 N(cfg.n_frames)로 나눠 비율로 만든다.
    if dwell_map is None:
        occupancy = np.full_like(density, np.nan, dtype=np.float32)
        dbg["dbg_occ_mode_none"] = 1.0
    else:
        dm = dwell_map.astype(np.float32)
        mx = float(np.nanmax(dm)) if np.isfinite(dm).any() else 0.0
        if mx <= 1.0 + 1e-6:
            occupancy = np.clip(dm, 0.0, 1.0)
            dbg["dbg_occ_mode_ratio"] = 1.0
        else:
            n_frames = float(max(1, int(cfg.n_frames)))
            occupancy = np.clip(dm / n_frames, 0.0, 1.0)
            dbg["dbg_occ_mode_count"] = 1.0
            dbg["dbg_occ_divN"] = float(n_frames)

    # 3) 동적 focus
    if dwell_map is not None:
        dm = dwell_map.astype(np.float32)
        focus_dyn = dm > 0
        dbg["dbg_focus_mode_dwell"] = 1.0
    else:
        focus_dyn = _topk_mask(density, cfg.topk_ratio_dyn)
        dbg["dbg_focus_mode_topk"] = 1.0

    # focus 셀 개수 기록
    dbg["dbg_focus_n"] = float(int(np.sum(focus_dyn)))

    if not focus_dyn.any():
        feats = {
            "density_mean": 0.0,
            "speed_mean": _safe_nanmean(mean_speed_map.astype(np.float32)) if mean_speed_map is not None else float("nan"),
            "occupancy_mean": 0.0 if dwell_map is not None else float("nan"),
            "ego_motion": float("nan"),
            "static_dwell_norm": float("nan"),
            "static_change_focus": float("nan"),
            "occlusion_score": 0.0,
            "congestion_score": 0.0,
            "speed_samples_mean": float("nan"),
            "speed_reliable_ratio": float("nan"),
            "ego_speed_mean": float("nan"),
            "speed_ratio": float("nan"),
            "speed_std_mean": float("nan"),
            "occlusion_like": False,
            "empty_like": True,
        }

        # 디버그: 왜 focus가 비었는지 빠르게 확인
        if debug:
            dbg.update(_dbg_stats("dbg_density_all", density, None))
            dbg.update(
                _dbg_stats(
                    "dbg_dwell_all",
                    dwell_map.astype(np.float32) if dwell_map is not None else None,
                    None,
                )
            )
            dbg["dbg_exit_no_focus"] = 1.0
            if dbg_print_on:
                _dbg_print(
                    True,
                    f"[ENC DBG]{(' ' + debug_tag) if debug_tag else ''} no focus -> Empty | "
                    f"focus_n={int(dbg['dbg_focus_n'])} dens_mean_all={dbg.get('dbg_density_all_mean')} "
                    f"dwell_mean_all={dbg.get('dbg_dwell_all_mean')}"
                )
            feats.update(dbg)
        return {}, ("Empty", feats)

    dens_mean = _safe_nanmean(density[focus_dyn])
    dbg["dbg_dens_mean_focus"] = float(dens_mean) if np.isfinite(dens_mean) else float("nan")

    roi_mean = float("nan")
    if roi_count_series is not None and roi_count_series.size > 0:
        roi_mean = float(np.mean(roi_count_series))

    # 1) ROI 차량 수 기반 empty (메인)
    empty_by_roi = np.isfinite(roi_mean) and (roi_mean < cfg.empty_roi_mean_thr)

    # 2) density 기반 empty (보조, roi_mean이 없을 때만 쓰거나 약하게)
    empty_by_dens = (not np.isfinite(roi_mean)) and np.isfinite(dens_mean) and (dens_mean < cfg.empty_dens_mean_thr)
    empty_like = bool(empty_by_roi or empty_by_dens)

    if empty_like:
        feats = {
            "density_mean": float(dens_mean) if np.isfinite(dens_mean) else 0.0,
            "speed_mean": float("nan"),
            "occupancy_mean": float("nan"),
            "ego_motion": float("nan"),
            "static_dwell_norm": float("nan"),
            "static_change_focus": float("nan"),
            "occlusion_score": 0.0,
            "congestion_score": 0.0,
            "speed_samples_mean": float("nan"),
            "speed_reliable_ratio": float("nan"),
            "ego_speed_mean": float("nan"),
            "speed_ratio": float("nan"),
            "speed_std_mean": float("nan"),
            "occlusion_like": False,
            "empty_like": True,
        }
        if debug:
            feats.update(dbg)
        return {}, ("Empty", feats)

    return {
        "density": density,
        "occupancy": occupancy,
        "focus_dyn": focus_dyn,
        "dens_mean": dens_mean,
    }, None


def _compute_speed_context(
    *,
    mean_speed_map: Optional[np.ndarray],
    std_speed_map: Optional[np.ndarray],
    speed_samples_map: Optional[np.ndarray],
    occupancy: np.ndarray,
    focus_dyn: np.ndarray,
    dwell_map: Optional[np.ndarray],
    cfg: EncoderConfig,
    debug: bool,
    dbg_print_on: bool,
    debug_tag: str,
    dbg: Dict[str, float],
) -> Dict[str, Any]:
    speed_mean = float("nan")
    speed_samples_mean = float("nan")
    speed_reliable_ratio = float("nan")
    speed_std_mean = float("nan")

    dbg["dbg_speed_has_samples_map"] = 1.0 if (speed_samples_map is not None) else 0.0
    dbg["dbg_speed_require_samples"] = 1.0 if bool(cfg.require_speed_samples) else 0.0
    dbg["dbg_min_speed_samples"] = float(cfg.min_speed_samples)

    reliable_cells_for_dbg: Optional[np.ndarray] = None
    speed_focus: Optional[np.ndarray] = None

    if speed_samples_map is not None:
        ss = speed_samples_map.astype(np.float32)
        focus_cells = focus_dyn
        focus_n = int(np.sum(focus_cells))
        dbg["dbg_focus_n_int"] = float(focus_n)

        if focus_n > 0:
            speed_samples_mean = float(np.nanmean(ss[focus_cells]))
            reliable_cells = focus_cells & (ss >= cfg.min_speed_samples)
            reliable_cells_for_dbg = reliable_cells
            reliable_n = int(np.sum(reliable_cells))
            speed_reliable_ratio = reliable_n / float(focus_n)

            dbg["dbg_reliable_n"] = float(reliable_n)
            dbg["dbg_speed_reliable_ratio"] = float(speed_reliable_ratio)

            if reliable_n > 0:
                speed_focus = _topk_mask_on_values(ss, reliable_cells, cfg.topk_ratio_speed)
                dbg["dbg_speed_focus_mode"] = 1.0
                dbg["dbg_speed_focus_ratio"] = float(cfg.topk_ratio_speed)
                dbg["dbg_speed_focus_n"] = float(int(np.sum(speed_focus)))
            else:
                speed_focus = reliable_cells
                dbg["dbg_speed_focus_mode"] = 0.0
                dbg["dbg_speed_focus_n"] = float(int(np.sum(speed_focus)))

            if mean_speed_map is not None:
                sp = mean_speed_map.astype(np.float32)
                use_mask = None
                if speed_focus is not None and np.any(speed_focus):
                    use_mask = speed_focus
                elif reliable_n > 0:
                    use_mask = reliable_cells
                else:
                    sampled_cells = focus_cells & (ss > 0)
                    use_mask = sampled_cells if np.any(sampled_cells) else None

                if use_mask is not None and np.any(use_mask):
                    speed_mean = _safe_nanmean(sp[use_mask])

            if std_speed_map is not None:
                st = std_speed_map.astype(np.float32)
                use_mask = speed_focus if (speed_focus is not None and np.any(speed_focus)) else reliable_cells
                if np.any(use_mask):
                    speed_std_mean = _safe_nanmean(st[use_mask])
                else:
                    dbg["dbg_speed_std_nomask"] = 1.0
            else:
                dbg["dbg_speed_std_map_missing"] = 1.0

            dbg.update(_dbg_stats("dbg_ss_focus", ss, focus_cells))
            dbg.update(_dbg_stats("dbg_ss_reliable", ss, reliable_cells))
            if speed_focus is not None:
                dbg.update(_dbg_stats("dbg_ss_speed_focus", ss, speed_focus))

            if debug and cfg.debug_speed_nan and (mean_speed_map is not None):
                sp = mean_speed_map.astype(np.float32)
                sp_focus = sp[focus_cells]
                finite_focus = np.isfinite(sp_focus)
                dbg["dbg_sp_focus_finite_n"] = float(int(np.sum(finite_focus)))
                dbg["dbg_sp_focus_finite_ratio"] = float(np.sum(finite_focus) / max(1, sp_focus.size))

                if speed_focus is not None and np.any(speed_focus):
                    sp_sf = sp[speed_focus]
                    finite_sf = np.isfinite(sp_sf)
                    dbg["dbg_sp_speed_focus_finite_n"] = float(int(np.sum(finite_sf)))
                    dbg["dbg_sp_speed_focus_finite_ratio"] = float(np.sum(finite_sf) / max(1, sp_sf.size))
        else:
            dbg["dbg_speed_focus_empty_focus"] = 1.0
    else:
        if mean_speed_map is not None:
            speed_mean = _safe_nanmean(mean_speed_map.astype(np.float32)[focus_dyn])
        if std_speed_map is not None:
            speed_std_mean = _safe_nanmean(std_speed_map.astype(np.float32)[focus_dyn])
        dbg["dbg_speed_no_samples_map_path"] = 1.0

    speed_valid = bool(np.isfinite(speed_mean))
    dbg["dbg_speed_mean_finite"] = 1.0 if np.isfinite(speed_mean) else 0.0

    if np.isfinite(speed_reliable_ratio):
        speed_valid = speed_valid and (speed_reliable_ratio >= cfg.speed_reliable_ratio_min)
        dbg["dbg_speed_ratio_gate"] = 1.0
        dbg["dbg_speed_ratio_min"] = float(cfg.speed_reliable_ratio_min)
    else:
        dbg["dbg_speed_ratio_gate"] = 0.0

    if not speed_valid:
        dbg["dbg_speed_valid"] = 0.0
        if not np.isfinite(speed_mean):
            dbg["dbg_speed_invalid_reason_nan"] = 1.0
        if np.isfinite(speed_reliable_ratio) and (speed_reliable_ratio < cfg.speed_reliable_ratio_min):
            dbg["dbg_speed_invalid_reason_ratio"] = 1.0

        if debug and cfg.debug_speed_nan:
            dbg.update(
                _dbg_speed_nan_diagnose(
                    mean_speed_map=mean_speed_map,
                    std_speed_map=std_speed_map,
                    speed_samples_map=speed_samples_map,
                    focus_dyn=focus_dyn,
                    reliable_cells=reliable_cells_for_dbg,
                    cfg=cfg,
                )
            )
            if dbg_print_on:
                _dbg_print(
                    True,
                    f"[ENC DBG SPEED]{(' ' + debug_tag) if debug_tag else ''} "
                    f"speed_valid=0 mean_speed_map_present={int(dbg.get('dbg_sp_has_mean_speed_map', 0))} "
                    f"focus_finite_ratio={dbg.get('dbg_sp_focus_finite_ratio')} "
                    f"reliable_n={dbg.get('dbg_sp_reliable_n')} rel_finite_ratio={dbg.get('dbg_sp_reliable_finite_ratio')}"
                )

        speed_mean = float("nan")
        speed_std_mean = float("nan")
    else:
        dbg["dbg_speed_valid"] = 1.0

    if cfg.require_speed_samples and (speed_samples_map is None):
        dbg["dbg_speed_invalid_reason_require_samples"] = 1.0
        speed_valid = False
        speed_mean = float("nan")
        speed_std_mean = float("nan")

    occ_mean = _safe_nanmean(occupancy[focus_dyn]) if dwell_map is not None else float("nan")
    dbg["dbg_occ_mean_focus"] = float(occ_mean) if np.isfinite(occ_mean) else float("nan")

    return {
        "speed_mean": speed_mean,
        "speed_samples_mean": speed_samples_mean,
        "speed_reliable_ratio": speed_reliable_ratio,
        "speed_std_mean": speed_std_mean,
        "speed_valid": speed_valid,
        "occ_mean": occ_mean,
    }


def _compute_static_context(
    *,
    static_change_rate: Optional[np.ndarray],
    static_dwell_map: Optional[np.ndarray],
    ego_speed_series: Optional[np.ndarray],
    speed_mean: float,
    cfg: EncoderConfig,
    dbg: Dict[str, float],
) -> Dict[str, Any]:
    ego_motion = float("nan")
    sd_mean = float("nan")
    scr_mean = float("nan")

    if static_change_rate is not None:
        scr = static_change_rate.astype(np.float32)

        if static_dwell_map is not None:
            sd_norm = _normalize_01(static_dwell_map)
            focus_stat = _topk_mask(sd_norm, cfg.topk_ratio_stat)

            if focus_stat.any():
                scr_mean = _safe_nanmean(scr[focus_stat])
                sd_mean = _safe_nanmean(sd_norm[focus_stat])
            else:
                scr_mean = _safe_nanmean(scr)
                sd_mean = _safe_nanmean(_normalize_01(static_dwell_map))
        else:
            scr_mean = _safe_nanmean(scr)

        ego_motion = scr_mean
        dbg["dbg_ego_motion"] = float(ego_motion) if np.isfinite(ego_motion) else float("nan")
        dbg["dbg_sd_mean"] = float(sd_mean) if np.isfinite(sd_mean) else float("nan")
        dbg["dbg_scr_mean"] = float(scr_mean) if np.isfinite(scr_mean) else float("nan")
    else:
        dbg["dbg_static_missing"] = 1.0

    ego_speed_mean = float("nan")
    if ego_speed_series is not None:
        es = ego_speed_series.astype(np.float32)
        finite = np.isfinite(es)
        if np.any(finite):
            ego_speed_mean = float(np.nanmean(es[finite]))

    if np.isfinite(speed_mean) and np.isfinite(ego_speed_mean) and ego_speed_mean > 0.3:
        speed_ratio = float(speed_mean / ego_speed_mean)
    else:
        speed_ratio = float("nan")

    dbg["dbg_ego_speed_mean"] = float(ego_speed_mean) if np.isfinite(ego_speed_mean) else float("nan")
    dbg["dbg_speed_ratio"] = float(speed_ratio) if np.isfinite(speed_ratio) else float("nan")

    return {
        "ego_motion": ego_motion,
        "sd_mean": sd_mean,
        "scr_mean": scr_mean,
        "ego_speed_mean": ego_speed_mean,
        "speed_ratio": speed_ratio,
    }


def _decide_event_type(
    *,
    dens_mean: float,
    speed_mean: float,
    occ_mean: float,
    ego_motion: float,
    sd_mean: float,
    scr_mean: float,
    speed_samples_mean: float,
    speed_reliable_ratio: float,
    ego_speed_mean: float,
    speed_ratio: float,
    speed_std_mean: float,
    speed_valid: bool,
    stopped_ratio: Optional[float],
    cfg: EncoderConfig,
    debug: bool,
    dbg_print_on: bool,
    debug_tag: str,
    dbg: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    occlusion_score = 0.0

    ego_low = np.isfinite(ego_motion) and (ego_motion <= cfg.ego_motion_low)
    dbg["dbg_ego_low"] = 1.0 if ego_low else 0.0
    dbg["dbg_ego_motion_low_thr"] = float(cfg.ego_motion_low)

    if ego_low:
        if np.isfinite(sd_mean) and (sd_mean >= cfg.sd_high):
            occlusion_score += 1.0
        if np.isfinite(scr_mean) and (scr_mean <= cfg.scr_low):
            occlusion_score += 1.0

        dyn_present_for_occ = np.isfinite(dens_mean) and (dens_mean >= cfg.dens_low)
        if dyn_present_for_occ and (not np.isfinite(speed_mean)):
            occlusion_score += 0.5

    congestion_score = 0.0
    if np.isfinite(dens_mean) and (dens_mean >= cfg.dens_cong):
        congestion_score += 1.0
    if speed_valid and np.isfinite(speed_mean) and (speed_mean <= cfg.v_low):
        congestion_score += 1.0
    if np.isfinite(occ_mean) and (occ_mean >= cfg.occ_high):
        congestion_score += 1.0

    if speed_valid and (stopped_ratio is not None):
        if stopped_ratio < cfg.stopped_ratio_high:
            congestion_score -= 1.0

    stop_and_go = False
    if speed_valid and (stopped_ratio is not None):
        if (stopped_ratio >= cfg.stopped_ratio_high) and np.isfinite(speed_std_mean) and (speed_std_mean >= cfg.speed_std_mix_high):
            stop_and_go = True

    dyn_present = np.isfinite(dens_mean) and (dens_mean >= cfg.dens_low)
    dbg["dbg_dyn_present"] = 1.0 if dyn_present else 0.0
    dbg["dbg_dens_low_thr"] = float(cfg.dens_low)

    occlusion_like = bool(ego_low and (occlusion_score >= 2.0))
    is_congestion = bool(congestion_score >= 2.0)
    is_stopngo = bool(stop_and_go)
    is_normal = bool(
        dyn_present
        and speed_valid
        and np.isfinite(speed_mean)
        and (speed_mean >= cfg.v_ok)
        and (not np.isfinite(occ_mean) or occ_mean < cfg.occ_high)
    )

    if is_congestion:
        event = "Congestion"
    elif is_stopngo:
        event = "StopAndGo"
    elif is_normal:
        event = "Normal"
    elif (not dyn_present) and (not occlusion_like):
        event = "Normal"
    else:
        event = "Unknown"

    feats = {
        "density_mean": float(dens_mean),
        "speed_mean": float(speed_mean) if np.isfinite(speed_mean) else float("nan"),
        "occupancy_mean": float(occ_mean) if np.isfinite(occ_mean) else float("nan"),
        "ego_motion": float(ego_motion) if np.isfinite(ego_motion) else float("nan"),
        "static_dwell_norm": float(sd_mean) if np.isfinite(sd_mean) else float("nan"),
        "static_change_focus": float(scr_mean) if np.isfinite(scr_mean) else float("nan"),
        "occlusion_score": float(occlusion_score),
        "congestion_score": float(congestion_score),
        "speed_samples_mean": float(speed_samples_mean) if np.isfinite(speed_samples_mean) else float("nan"),
        "speed_reliable_ratio": float(speed_reliable_ratio) if np.isfinite(speed_reliable_ratio) else float("nan"),
        "ego_speed_mean": float(ego_speed_mean) if np.isfinite(ego_speed_mean) else float("nan"),
        "speed_ratio": float(speed_ratio) if np.isfinite(speed_ratio) else float("nan"),
        "speed_std_mean": float(speed_std_mean) if np.isfinite(speed_std_mean) else float("nan"),
        "occlusion_like": bool(occlusion_like),
        "empty_like": False,
    }

    if debug:
        feats.update(dbg)
        if dbg_print_on:
            _dbg_print(
                True,
                f"[ENC DBG]{(' ' + debug_tag) if debug_tag else ''} event={event} "
                f"focus_n={int(dbg.get('dbg_focus_n', 0))} dens={feats.get('density_mean')} "
                f"speed_mean={feats.get('speed_mean')} speed_valid={int(dbg.get('dbg_speed_valid', 0))} "
                f"reliable_ratio={feats.get('speed_reliable_ratio')} "
                f"ego_motion={feats.get('ego_motion')} occ={feats.get('occupancy_mean')} "
                f"occ_score={feats.get('occlusion_score')} cong_score={feats.get('congestion_score')}"
            )

    return event, feats


def encode_event_type(
    sum_count_map: Optional[np.ndarray] = None,
    mean_count_map: Optional[np.ndarray] = None,
    unique_cnt_map: Optional[np.ndarray] = None,
    mean_speed_map: Optional[np.ndarray] = None,
    std_speed_map: Optional[np.ndarray] = None,
    dwell_map: Optional[np.ndarray] = None,
    *,
    cfg: Optional[EncoderConfig] = None,
    static_dwell_map: Optional[np.ndarray] = None,
    static_change_rate: Optional[np.ndarray] = None,
    speed_samples_map: Optional[np.ndarray] = None,
    roi_count_series: Optional[np.ndarray] = None,
    ego_speed_series: Optional[np.ndarray] = None,
    stopped_ratio: Optional[float] = None,
    debug: Optional[bool] = None,
    debug_tag: str = "",
) -> Tuple[str, Dict[str, float]]:
    if debug is None:
        env_flag = str(os.getenv("EVENT_ENC_DEBUG", "")).strip().lower()
        env_debug = env_flag in ("1", "true", "yes", "y", "on")
        debug = bool(cfg.debug or env_debug)

    if cfg is None:
        cfg = EncoderConfig()

    dbg_print_on = bool(debug and cfg.debug_print)
    dbg: Dict[str, float] = {}

    density_ctx, early_result = _compute_density_context(
        sum_count_map=sum_count_map,
        mean_count_map=mean_count_map,
        unique_cnt_map=unique_cnt_map,
        dwell_map=dwell_map,
        mean_speed_map=mean_speed_map,
        speed_samples_map=speed_samples_map,
        static_change_rate=static_change_rate,
        static_dwell_map=static_dwell_map,
        roi_count_series=roi_count_series,
        cfg=cfg,
        debug=bool(debug),
        dbg_print_on=dbg_print_on,
        debug_tag=debug_tag,
        dbg=dbg,
    )
    if early_result is not None:
        return early_result

    speed_ctx = _compute_speed_context(
        mean_speed_map=mean_speed_map,
        std_speed_map=std_speed_map,
        speed_samples_map=speed_samples_map,
        occupancy=density_ctx["occupancy"],
        focus_dyn=density_ctx["focus_dyn"],
        dwell_map=dwell_map,
        cfg=cfg,
        debug=bool(debug),
        dbg_print_on=dbg_print_on,
        debug_tag=debug_tag,
        dbg=dbg,
    )

    static_ctx = _compute_static_context(
        static_change_rate=static_change_rate,
        static_dwell_map=static_dwell_map,
        ego_speed_series=ego_speed_series,
        speed_mean=speed_ctx["speed_mean"],
        cfg=cfg,
        dbg=dbg,
    )

    return _decide_event_type(
        dens_mean=density_ctx["dens_mean"],
        speed_mean=speed_ctx["speed_mean"],
        occ_mean=speed_ctx["occ_mean"],
        ego_motion=static_ctx["ego_motion"],
        sd_mean=static_ctx["sd_mean"],
        scr_mean=static_ctx["scr_mean"],
        speed_samples_mean=speed_ctx["speed_samples_mean"],
        speed_reliable_ratio=speed_ctx["speed_reliable_ratio"],
        ego_speed_mean=static_ctx["ego_speed_mean"],
        speed_ratio=static_ctx["speed_ratio"],
        speed_std_mean=speed_ctx["speed_std_mean"],
        speed_valid=speed_ctx["speed_valid"],
        stopped_ratio=stopped_ratio,
        cfg=cfg,
        debug=bool(debug),
        dbg_print_on=dbg_print_on,
        debug_tag=debug_tag,
        dbg=dbg,
    )


def encode_event(
    *,
    dwell_map: np.ndarray,
    mean_speed_map: np.ndarray,
    std_speed_map: np.ndarray,
    speed_samples_map: np.ndarray,
    static_dwell_map: np.ndarray,
    static_change_rate: np.ndarray,
    cfg: EncoderConfig,
    roi_count_series: Optional[np.ndarray] = None,
    ego_speed_series: Optional[np.ndarray] = None,
    stopped_ratio: Optional[float] = None,
    unique_cnt_map: Optional[np.ndarray] = None,
) -> Tuple[str, Dict[str, float]]:
    return encode_event_type(
        sum_count_map=None,
        mean_count_map=None,
        unique_cnt_map=unique_cnt_map,
        mean_speed_map=mean_speed_map,
        std_speed_map=std_speed_map,
        dwell_map=dwell_map,
        cfg=cfg,
        static_dwell_map=static_dwell_map,
        static_change_rate=static_change_rate,
        speed_samples_map=speed_samples_map,
        roi_count_series=roi_count_series,
        ego_speed_series=ego_speed_series,
        stopped_ratio=stopped_ratio,
        debug=None,
        debug_tag="",
    )
