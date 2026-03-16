# src/event_encoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

# (추가) 디버그 출력/환경변수 등을 쓰고 싶을 때 대비
import os


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

    speed_std_mix_high: float = 2.0     # stop-and-go(혼재) 판단 기준

    # ---- Focus(상위 k% 셀만 요약) ----
    topk_ratio: Optional[float] = None
    topk_ratio_dyn: float = 0.05
    topk_ratio_stat: float = 0.05

    # (추가/수정) Speed Focus(속도 계산은 dwell 기반 focus_dyn이 아니라 "샘플이 많은 셀"로 따로 잡는다)
    # - dwell 기반 focus는 "점유/혼잡"엔 좋지만, 속도는 track이 안정적으로 쌓인 셀만 써야 NaN이 줄어듦
    # - 0.05~0.20 범위 권장 (데이터/ROI에 따라 조절)
    topk_ratio_speed: float = 0.10

    # ---- Static(ego/occlusion) gating & thresholds ----
    ego_motion_low: float = 0.10

    # Occlusion 판정 (정적 “고정 패턴”)
    scr_low: float = 0.10
    sd_high: float = 0.60

    # density 기준
    dens_low: float = 0.20
    dens_cong: float = 0.60
    min_valid_density: float = 1e-6

    # speed 샘플 기반 유효성(있으면 훨씬 안정적)
    min_speed_samples: int = 1  # cnt_v >= 5인 셀만 속도 신뢰
    require_speed_samples: bool = True  # speed_samples_map 들어오면 샘플로 gate

    # (추가) 디버그 플래그
    # - cfg.debug=True면 encode_event_type 내부 중간값들을 print로 찍어줌
    # - cfg.debug_print=False면 feats에 dbg_* 키만 추가하고 print는 하지 않도록 할 수도 있음
    debug: bool = False
    debug_print: bool = True

    # (추가) speed NaN 원인추적 디버그(무거울 수 있음)
    # - True면 mean_speed_map 자체/마스크별 finite 비율 등을 더 자세히 기록
    debug_speed_nan: bool = True

    empty_roi_mean_thr: float = 0.5     # 윈도우 평균 ROI 차량수 < 이 값이면 "거의 없음"
    empty_dens_mean_thr: float = 0.05   # dens_mean < 이 값이면 "거의 없음"

    def __post_init__(self):
        if self.topk_ratio is not None:
            r = float(self.topk_ratio)
            self.topk_ratio_dyn = r
            self.topk_ratio_stat = r
            # (추가) topk_ratio를 주면 speed focus 비율도 같이 따라가게 하고 싶다면 아래를 켜도 됨
            # self.topk_ratio_speed = r


def _safe_nanmean(x: np.ndarray) -> float:
    if x is None:
        return float("nan")
    if x.size == 0:
        return float("nan")
    finite = np.isfinite(x)
    if not finite.any():
        return float("nan")
    return float(np.nanmean(x[finite]))


def _normalize_01(x: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float32)
    finite = np.isfinite(xf)
    if not finite.any():
        return np.zeros_like(xf, dtype=np.float32)
    mx = float(np.nanmax(xf[finite]))
    if mx <= 0:
        return np.zeros_like(xf, dtype=np.float32)
    return np.clip(xf / mx, 0.0, 1.0)


def _topk_mask(score_map: np.ndarray, ratio: float) -> np.ndarray:
    s = score_map.astype(np.float32).copy()
    s[~np.isfinite(s)] = np.nan
    flat = s[np.isfinite(s) & (s > 0)]
    if flat.size == 0:
        return np.zeros_like(s, dtype=bool)
    ratio = float(np.clip(ratio, 1e-6, 1.0))
    k = max(1, int(np.ceil(flat.size * ratio)))
    thr = np.partition(flat, -k)[-k]
    return np.isfinite(s) & (s > 0) & (s >= thr)


# (추가) speed focus를 만들기 위한 top-k 유틸 (샘플수 맵 같은 "값" 기반)
# - score_map의 (mask 내부) 값 중 상위 ratio만 True로 뽑아줌
def _topk_mask_on_values(score_map: np.ndarray, base_mask: np.ndarray, ratio: float) -> np.ndarray:
    s = score_map.astype(np.float32)
    m = base_mask.astype(bool)

    vals = s[m]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(m, dtype=bool)

    ratio = float(np.clip(ratio, 1e-6, 1.0))
    thr = float(np.quantile(vals, 1.0 - ratio))
    return m & np.isfinite(s) & (s >= thr)


def _density_from_sumcount(sum_count_map: np.ndarray, cfg: EncoderConfig) -> np.ndarray:
    sc = sum_count_map.astype(np.float32)
    N = max(1, int(cfg.n_frames))
    mean_count_cell = sc / float(N)
    return np.clip(mean_count_cell / float(cfg.density_cap_cell), 0.0, 1.0)


def _density_from_meancount(mean_count_map: np.ndarray, cfg: EncoderConfig) -> np.ndarray:
    mc = mean_count_map.astype(np.float32)
    return np.clip(mc / float(cfg.density_cap_cell), 0.0, 1.0)


# (추가) 디버그 프린트 헬퍼
def _dbg_print(enabled: bool, msg: str):
    """디버그 출력 스위치"""
    if enabled:
        print(msg)


# (추가) 안전한 통계 요약: 배열의 min/max/mean 같은 것들을 한 번에 보기 위함
def _dbg_stats(name: str, arr: Optional[np.ndarray], mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    디버깅용 통계 요약.
    - arr이 None이면 nan으로 채움
    - mask가 있으면 mask True인 영역만 집계
    """
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
            # mask shape mismatch 같은 예외가 나면 전체로 fallback
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


# (추가) speed_mean이 NaN으로 떨어지는 "원인"을 더 직접적으로 잡아내는 진단 함수
def _dbg_speed_nan_diagnose(
    *,
    mean_speed_map: Optional[np.ndarray],
    std_speed_map: Optional[np.ndarray],
    speed_samples_map: Optional[np.ndarray],
    focus_dyn: np.ndarray,
    reliable_cells: Optional[np.ndarray],
    cfg: EncoderConfig,
) -> Dict[str, float]:
    """
    speed_mean NaN 디버깅용.
    - "샘플은 있는데(mean cnt_v는 있는데) mean_speed_map이 전부 NaN" 같은 케이스를 잡아냄
    - 마스크별 finite 비율/개수/범위 등을 feats에 담아 로그로 남김
    """
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
        # focus에서 ss>=min_speed_samples인 셀 수
        rel = (focus_dyn & (ss >= cfg.min_speed_samples))
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


def encode_event_type(
    sum_count_map: Optional[np.ndarray] = None,
    mean_count_map: Optional[np.ndarray] = None,
    unique_cnt_map: Optional[np.ndarray] = None,

    mean_speed_map: Optional[np.ndarray] = None,
    std_speed_map: Optional[np.ndarray] = None,
    dwell_map: Optional[np.ndarray] = None,

    *,
    cfg: EncoderConfig = EncoderConfig(),
    static_dwell_map: Optional[np.ndarray] = None,
    static_change_rate: Optional[np.ndarray] = None,

    # speed 샘플 수 맵 (track_heatmaps의 cnt_v)
    speed_samples_map: Optional[np.ndarray] = None,

    roi_count_series: Optional[np.ndarray] = None,  # (N,) 프레임별 ROI 차량 수

    # pose 없이 추정한 ego 속도 시계열 (EventWindow에서 static_occ shift로 계산)
    ego_speed_series: Optional[np.ndarray] = None,

    stopped_ratio: Optional[float] = None,

    # (추가) 호출부에서 cfg.debug를 건드리기 싫으면 여기로도 켤 수 있게
    debug: Optional[bool] = None,
    debug_tag: str = "",  # 디버그 출력에 붙일 라벨(예: "WIN t=300")
) -> Tuple[str, Dict[str, float]]:
    """
    안정화 버전:
    - Occlusion은 "ego_motion(정적 변화) 낮음"이 반드시 필요하도록 강화
    - '차가 없어서 speed가 NaN' 같은 케이스가 Occlusion로 올라가지 않도록 억제
    - speed_samples_map이 들어오면 속도 신뢰도(샘플 부족)를 구분

    (추가/수정) 속도 NaN 완화:
    - dwell 기반 focus_dyn(점유>0)에서 속도를 평균내면,
      speed_samples가 없는 셀(=추적/매칭 실패 셀) 때문에 mean_speed_map이 전부 NaN이 되는 구간이 생김.
    - 그래서 속도는 "샘플이 있는 셀"에서만, 더 나아가 "샘플이 많은 상위 k%"(topk_ratio_speed)로 평균냄.
    """

    # (추가) 디버그 on/off 결정
    # - 우선순위: encode_event_type(debug=...) 인자 > cfg.debug > 환경변수 EVENT_ENC_DEBUG
    # - 환경변수는 1/true/yes 를 True로 취급
    if debug is None:
        env_flag = str(os.getenv("EVENT_ENC_DEBUG", "")).strip().lower()
        env_debug = env_flag in ("1", "true", "yes", "y", "on")
        debug = bool(cfg.debug or env_debug)
    dbg_print_on = bool(debug and cfg.debug_print)

    # (추가) 디버그 정보 누적(출력/feats에 넣기)
    dbg: Dict[str, float] = {}

    # ---- 기준 shape 결정 (H,W) 확보 ----
    ref_shape = None
    for m in (dwell_map, mean_speed_map, speed_samples_map, static_change_rate, static_dwell_map, unique_cnt_map,
              sum_count_map, mean_count_map):
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
        # (추가) 디버그: ROI 기반 density 정보
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
                0.0, 1.0
            )
            dbg["dbg_density_mode_unique"] = 1.0

        elif dwell_map is not None:
            # 최후 fallback: dwell을 0~1 정규화해서 density 대용
            density = _normalize_01(dwell_map.astype(np.float32))
            dbg["dbg_density_mode_dwell_fallback"] = 1.0

        else:
            # 여기까지 오면 진짜로 아무 정보가 없는 거라 에러가 맞음
            raise ValueError(
                "Need one of {roi_count_series, sum_count_map, mean_count_map, unique_cnt_map, dwell_map}.")

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
            # 이미 비율(0~1)로 들어온 경우 (예: event_window에서 /N 해서 넘김)
            occupancy = np.clip(dm, 0.0, 1.0)
            dbg["dbg_occ_mode_ratio"] = 1.0
        else:
            # 카운트로 들어온 경우 (예: 0~N 프레임 누적)
            N = float(max(1, int(cfg.n_frames)))
            occupancy = np.clip(dm / N, 0.0, 1.0)
            dbg["dbg_occ_mode_count"] = 1.0
            dbg["dbg_occ_divN"] = float(N)

    # 3) 동적 focus
    if dwell_map is not None:
        dm = dwell_map.astype(np.float32)
        focus_dyn = (dm > 0)  # 또는 dm >= (1.0 / cfg.n_frames) 같은 기준
        dbg["dbg_focus_mode_dwell"] = 1.0
    else:
        focus_dyn = _topk_mask(density, cfg.topk_ratio_dyn)
        dbg["dbg_focus_mode_topk"] = 1.0

    # (추가) focus 셀 개수 기록
    dbg["dbg_focus_n"] = float(int(np.sum(focus_dyn)))

    # 동적이 거의 없으면 Normal
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
        }

        # (추가) 디버그: 왜 focus가 비었는지 빠르게 확인
        if debug:
            # dwell_map이 있는데도 focus가 비면 (=dwell_map이 전부 0)
            # 또는 topk가 전부 0/NaN인 경우가 대부분
            dbg.update(_dbg_stats("dbg_density_all", density, None))
            dbg.update(_dbg_stats("dbg_dwell_all", dwell_map.astype(np.float32) if dwell_map is not None else None, None))
            dbg["dbg_exit_no_focus"] = 1.0
            if dbg_print_on:
                _dbg_print(True, f"[ENC DBG]{(' ' + debug_tag) if debug_tag else ''} no focus -> Normal | "
                                 f"focus_n={int(dbg['dbg_focus_n'])} dens_mean_all={dbg.get('dbg_density_all_mean')} "
                                 f"dwell_mean_all={dbg.get('dbg_dwell_all_mean')}")
            feats.update(dbg)
        return "Empty", feats

    dens_mean = _safe_nanmean(density[focus_dyn])
    dbg["dbg_dens_mean_focus"] = float(dens_mean) if np.isfinite(dens_mean) else float("nan")

    # ==========================================================
    # [PATCH] Empty scene early-return
    # - 차량이 거의 없으면 speed/occlusion 로직이 의미가 없어지고 NaN이 연쇄됨
    # - 이 케이스는 무조건 Normal로 보냄 (+ empty_like=1 플래그)
    # ==========================================================
    roi_mean = float("nan")
    if roi_count_series is not None and roi_count_series.size > 0:
        roi_mean = float(np.mean(roi_count_series))

    # 1) ROI 차량 수 기반 empty (메인)
    empty_by_roi = (np.isfinite(roi_mean) and roi_mean < cfg.empty_roi_mean_thr)

    # 2) density 기반 empty (보조, roi_mean이 없을 때만 쓰거나 약하게)
    empty_by_dens = (not np.isfinite(roi_mean)) and (np.isfinite(dens_mean) and dens_mean < cfg.empty_dens_mean_thr)

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
            "empty_like": True,  # 유지
        }
        if debug:
            feats.update(dbg)
        return "Empty", feats

    # 3-1) speed 유효성: "평균 샘플수"로 gate 하지 말고,
    #      셀 단위로 (cnt_v >= min_speed_samples)인 것만 평균내야 함.
    speed_mean = float("nan")
    speed_samples_mean = float("nan")
    speed_reliable_ratio = float("nan")
    speed_std_mean = float("nan")

    # (추가) speed_valid 실패 이유 추적용 플래그들(0/1로 넣어서 로깅하기 좋게)
    dbg["dbg_speed_has_samples_map"] = 1.0 if (speed_samples_map is not None) else 0.0
    dbg["dbg_speed_require_samples"] = 1.0 if bool(cfg.require_speed_samples) else 0.0
    dbg["dbg_min_speed_samples"] = float(cfg.min_speed_samples)

    # (추가) "speed_mean이 NaN이 되는" 케이스를 더 정확히 분해하기 위한 변수
    reliable_cells_for_dbg: Optional[np.ndarray] = None

    # (추가/수정) 속도 계산용 focus (speed_focus) 생성:
    # - dwell 기반 focus_dyn 안에서, speed_samples_map(cnt_v)이 충분한 셀만 대상으로
    # - 그 중에서도 샘플이 많은 상위 topk_ratio_speed만 평균에 사용
    speed_focus: Optional[np.ndarray] = None

    if speed_samples_map is not None:
        ss = speed_samples_map.astype(np.float32)

        # focus 영역에서 "속도 샘플이 있는 셀"만 먼저 집계
        focus_cells = focus_dyn
        focus_n = int(np.sum(focus_cells))
        dbg["dbg_focus_n_int"] = float(focus_n)

        if focus_n > 0:
            speed_samples_mean = float(np.nanmean(ss[focus_cells]))  # 참고용(그대로 둬도 됨)

            # 신뢰 가능한 셀 마스크(최소 샘플 수)
            reliable_cells = focus_cells & (ss >= cfg.min_speed_samples)
            reliable_cells_for_dbg = reliable_cells  # (추가) speed NaN 진단용으로 저장
            reliable_n = int(np.sum(reliable_cells))
            speed_reliable_ratio = reliable_n / float(focus_n)

            # (추가) 디버그: reliable 셀 개수/비율
            dbg["dbg_reliable_n"] = float(reliable_n)
            dbg["dbg_speed_reliable_ratio"] = float(speed_reliable_ratio)

            # (추가/수정) speed_focus 생성:
            # - reliable_cells(샘플>=min) 중에서 샘플이 많은 상위 k%를 사용
            # - 이게 핵심: dwell focus 때문에 "속도 없는 셀" 섞여서 NaN 되는 걸 줄임
            if reliable_n > 0:
                speed_focus = _topk_mask_on_values(ss, reliable_cells, cfg.topk_ratio_speed)
                dbg["dbg_speed_focus_mode"] = 1.0
                dbg["dbg_speed_focus_ratio"] = float(cfg.topk_ratio_speed)
                dbg["dbg_speed_focus_n"] = float(int(np.sum(speed_focus)))
            else:
                speed_focus = reliable_cells  # (전부 False일 수 있음)
                dbg["dbg_speed_focus_mode"] = 0.0
                dbg["dbg_speed_focus_n"] = float(int(np.sum(speed_focus)))

            # (추가/수정) speed_mean 계산:
            # - 우선 speed_focus에서 평균(샘플 많은 셀)
            # - speed_focus가 비었으면 reliable_cells로 fallback
            if mean_speed_map is not None:
                sp = mean_speed_map.astype(np.float32)

                use_mask = None
                if speed_focus is not None and np.any(speed_focus):
                    use_mask = speed_focus
                elif reliable_n > 0:
                    use_mask = reliable_cells
                else:
                    # 신뢰셀 없으면 샘플 있는 셀로 fallback
                    sampled_cells = focus_cells & (ss > 0)
                    use_mask = sampled_cells if np.any(sampled_cells) else None

                if use_mask is not None and np.any(use_mask):
                    speed_mean = _safe_nanmean(sp[use_mask])
                else:
                    speed_mean = float("nan")

            # (추가/수정) std_speed_map도 동일 use_mask로 평균
            if std_speed_map is not None:
                st = std_speed_map.astype(np.float32)
                use_mask = speed_focus if (speed_focus is not None and np.any(speed_focus)) else reliable_cells

                if np.any(use_mask):
                    st_use = st[use_mask]
                    speed_std_mean = _safe_nanmean(st_use)
                else:
                    speed_std_mean = float("nan")
                    dbg["dbg_speed_std_nomask"] = 1.0
            else:
                speed_std_mean = float("nan")
                dbg["dbg_speed_std_map_missing"] = 1.0

            # (추가) 디버그: 샘플 수 자체의 분포를 보면 왜 reliable이 0인지 바로 보임
            dbg.update(_dbg_stats("dbg_ss_focus", ss, focus_cells))
            dbg.update(_dbg_stats("dbg_ss_reliable", ss, reliable_cells))
            if speed_focus is not None:
                dbg.update(_dbg_stats("dbg_ss_speed_focus", ss, speed_focus))

            # (추가) 디버그: mean_speed_map 자체가 focus에서 얼마나 비어있는지(전체적인 NaN 상태)
            if debug and cfg.debug_speed_nan and (mean_speed_map is not None):
                sp = mean_speed_map.astype(np.float32)
                sp_focus = sp[focus_cells]
                finite_focus = np.isfinite(sp_focus)
                dbg["dbg_sp_focus_finite_n"] = float(int(np.sum(finite_focus)))
                dbg["dbg_sp_focus_finite_ratio"] = float(np.sum(finite_focus) / max(1, sp_focus.size))

                # (추가) speed_focus 기준으로도 finite 비율 확인 (원인 추적에 더 직접적)
                if speed_focus is not None and np.any(speed_focus):
                    sp_sf = sp[speed_focus]
                    finite_sf = np.isfinite(sp_sf)
                    dbg["dbg_sp_speed_focus_finite_n"] = float(int(np.sum(finite_sf)))
                    dbg["dbg_sp_speed_focus_finite_ratio"] = float(np.sum(finite_sf) / max(1, sp_sf.size))
        else:
            # focus 자체가 0
            speed_mean = float("nan")
            speed_std_mean = float("nan")
            dbg["dbg_speed_focus_empty_focus"] = 1.0
    else:
        # speed_samples_map이 없으면 그냥 focus 평균
        if mean_speed_map is not None:
            sp = mean_speed_map.astype(np.float32)
            speed_mean = _safe_nanmean(sp[focus_dyn])

        if std_speed_map is not None:
            st = std_speed_map.astype(np.float32)
            speed_std_mean = _safe_nanmean(st[focus_dyn])

        # (추가) 디버그: 샘플 맵 없이 들어왔을 때 표시
        dbg["dbg_speed_no_samples_map_path"] = 1.0

    # Speed validity gate (있을 때만 사용)
    speed_valid = bool(np.isfinite(speed_mean))
    dbg["dbg_speed_mean_finite"] = 1.0 if np.isfinite(speed_mean) else 0.0

    if np.isfinite(speed_reliable_ratio):
        speed_valid = speed_valid and (speed_reliable_ratio >= cfg.speed_reliable_ratio_min)
        dbg["dbg_speed_ratio_gate"] = 1.0
        dbg["dbg_speed_ratio_min"] = float(cfg.speed_reliable_ratio_min)
    else:
        dbg["dbg_speed_ratio_gate"] = 0.0

    if not speed_valid:
        # (추가) speed_valid가 False로 떨어질 때의 원인을 feats로 남김
        dbg["dbg_speed_valid"] = 0.0
        if not np.isfinite(speed_mean):
            dbg["dbg_speed_invalid_reason_nan"] = 1.0
        if np.isfinite(speed_reliable_ratio) and (speed_reliable_ratio < cfg.speed_reliable_ratio_min):
            dbg["dbg_speed_invalid_reason_ratio"] = 1.0

        # (추가) speed_mean NaN 원인을 더 강하게 파헤침(핵심!)
        # - "샘플은 존재(reliable_n>0)하는데 mean_speed_map이 reliable 영역에서 전부 NaN"인지 확인 가능
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
        # require_speed_samples가 True인데 speed_samples_map을 안 준 경우: 강제로 invalid 처리
        dbg["dbg_speed_invalid_reason_require_samples"] = 1.0
        speed_valid = False
        speed_mean = float("nan")
        speed_std_mean = float("nan")

    occ_mean = _safe_nanmean(occupancy[focus_dyn]) if dwell_map is not None else float("nan")
    dbg["dbg_occ_mean_focus"] = float(occ_mean) if np.isfinite(occ_mean) else float("nan")

    # 4) 정적 요약 (ego_motion)
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

        # (추가) 디버그: 정적 요약 통계
        dbg["dbg_ego_motion"] = float(ego_motion) if np.isfinite(ego_motion) else float("nan")
        dbg["dbg_sd_mean"] = float(sd_mean) if np.isfinite(sd_mean) else float("nan")
        dbg["dbg_scr_mean"] = float(scr_mean) if np.isfinite(scr_mean) else float("nan")
    else:
        dbg["dbg_static_missing"] = 1.0

    # 4-1) ego speed (pose 없이) 요약
    ego_speed_mean = float("nan")
    if ego_speed_series is not None:
        es = ego_speed_series.astype(np.float32)
        finite = np.isfinite(es)
        if np.any(finite):
            ego_speed_mean = float(np.nanmean(es[finite]))

    # speed_ratio: ego_speed에 비해 주변차 상대속도가 얼마나 되나(0에 가까울수록 "같이 흐름")
    if np.isfinite(speed_mean) and np.isfinite(ego_speed_mean) and ego_speed_mean > 0.3:
        speed_ratio = float(speed_mean / ego_speed_mean)
    else:
        speed_ratio = float("nan")

    # (추가) 디버그: ego_speed 관련
    dbg["dbg_ego_speed_mean"] = float(ego_speed_mean) if np.isfinite(ego_speed_mean) else float("nan")
    dbg["dbg_speed_ratio"] = float(speed_ratio) if np.isfinite(speed_ratio) else float("nan")

    # ------------------------------------------------------------------
    # 5) Occlusion score (강화)
    # 핵심: ego_motion이 낮지 않으면(=움직임이 있으면) Occlusion로 확정하지 않음
    # ------------------------------------------------------------------
    occlusion_score = 0.0

    # (필수 게이트) ego_motion이 낮아야 occlusion 후보
    ego_low = (np.isfinite(ego_motion) and ego_motion <= cfg.ego_motion_low)
    dbg["dbg_ego_low"] = 1.0 if ego_low else 0.0
    dbg["dbg_ego_motion_low_thr"] = float(cfg.ego_motion_low)

    if ego_low:
        # 정적 고정 패턴 강함: dwell↑ + change↓
        if np.isfinite(sd_mean) and sd_mean >= cfg.sd_high:
            occlusion_score += 1.0
        if np.isfinite(scr_mean) and scr_mean <= cfg.scr_low:
            occlusion_score += 1.0

        # "차가 있는데 속도만 비는" 이상 징후 보조 (차가 거의 없으면 occlusion로 올리지 않음)
        dyn_present = (np.isfinite(dens_mean) and dens_mean >= cfg.dens_low)
        if dyn_present and (not np.isfinite(speed_mean)):
            occlusion_score += 0.5

    # ------------------------------------------------------------------
    # 6) Congestion score
    # ------------------------------------------------------------------
    congestion_score = 0.0
    if np.isfinite(dens_mean) and dens_mean >= cfg.dens_cong:
        congestion_score += 1.0
    if speed_valid and np.isfinite(speed_mean) and speed_mean <= cfg.v_low:
        congestion_score += 1.0
    if np.isfinite(occ_mean) and occ_mean >= cfg.occ_high:
        congestion_score += 1.0

    # "정차 차량이 소수"면 정체 과판정 방지
    if speed_valid and (stopped_ratio is not None):
        if stopped_ratio < cfg.stopped_ratio_high:
            congestion_score -= 1.0

    stop_and_go = False
    if speed_valid and (stopped_ratio is not None):
        if (stopped_ratio >= cfg.stopped_ratio_high and
                np.isfinite(speed_std_mean) and speed_std_mean >= cfg.speed_std_mix_high):
            stop_and_go = True

    # ------------------------------------------------------------------
    # 7) 최종 결정
    # ------------------------------------------------------------------
    # 동적 존재 여부(차가 좀 있어야 교통상태를 판단할 의미가 있음)
    dyn_present = (np.isfinite(dens_mean) and dens_mean >= cfg.dens_low)
    dbg["dbg_dyn_present"] = 1.0 if dyn_present else 0.0
    dbg["dbg_dens_low_thr"] = float(cfg.dens_low)

    # 관측/품질이 애매한 상황(= Occlusion-like). "클래스"로 쓰지 않고 Unknown으로 보냄
    occlusion_like = bool(ego_low and occlusion_score >= 2.0)

    # 교통상태를 "확신"할 조건들
    is_congestion = bool(congestion_score >= 2.0)
    is_stopngo = bool(stop_and_go)

    # Normal을 "확신"하려면: 차가 너무 적지 않고 + 속도도 정상 범위 + 점유 낮음
    is_normal = bool(
        dyn_present and
        speed_valid and np.isfinite(speed_mean) and (speed_mean >= cfg.v_ok) and
        (not np.isfinite(occ_mean) or occ_mean < cfg.occ_high)
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
        # 어떤 교통상태도 확신 못 하면 Unknown
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
    }

    # (추가) 디버그: feats에 내부 중간값을 같이 실어보내기 (로그/저장 시 같이 남아서 원인추적 쉬움)
    if debug:
        feats.update(dbg)

        # (추가) 디버그 print: 한 줄 요약 (윈도우/프레임별로 봐야 할 핵심들만)
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


# event_window.py에서 import 하는 이름(호환용)
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
    # std_speed_map은 현재 encode_event_type에서 직접 쓰진 않지만,
    # 인터페이스 유지(필요하면 이후 congestion/stop&go 등에 추가 활용 가능)
    return encode_event_type(
        sum_count_map=None,
        mean_count_map=None,
        unique_cnt_map=unique_cnt_map,
        mean_speed_map=mean_speed_map,
        dwell_map=dwell_map,
        cfg=cfg,
        static_dwell_map=static_dwell_map,
        static_change_rate=static_change_rate,
        speed_samples_map=speed_samples_map,
        roi_count_series=roi_count_series,
        ego_speed_series=ego_speed_series,
        stopped_ratio=stopped_ratio,
        std_speed_map=std_speed_map,

        # (추가) encode_event 쪽에서도 cfg.debug를 그대로 타게 둠
        # debug 인자를 따로 넘기고 싶으면 encode_event_type를 직접 호출하면 됨
        debug=None,
        debug_tag="",
    )
