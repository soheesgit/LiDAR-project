# heatmap_writer.py
# mean speed, std, dwell, class mask 같은 분석 결과를 PNG 히트맵으로 저장

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def auto_range(arr: np.ndarray):
    """NaN/inf 제외한 자동 범위 계산"""
    finite = np.isfinite(arr)           # 유한한 값(True) / NaN, inf(False) 마스크
    if not finite.any():                # 유효한 값이 하나도 없으면
        return None, None               # (None, None) 반환

    lo = float(np.nanmin(arr[finite]))  # 배열 내 최소값
    hi = float(np.nanmax(arr[finite]))  # 배열 내 최대값

    if lo == hi:                        # min=max라서 범위가 0이면
        hi = lo + 1e-6                  # hi를 살짝 늘려서 (시각화 에러 방지)

    return lo, hi                       # (최솟값, 최댓값) 반환


def save_map(path: Path, title: str, arr: np.ndarray, vmin=None, vmax=None):
    """히트맵 저장 (vmin/vmax 지정 가능)"""
    path.parent.mkdir(parents=True, exist_ok=True)      # 저장 경로 없으면 생성

    plt.figure()                                        # 새 Figure 시작
    if vmin is None or vmax is None:                    # min/max 지정 안 했을 경우
        vmin2, vmax2 = auto_range(arr)                  # auto_range로 범위 계산
        vmin = vmin if vmin is not None else vmin2      # 지정값이 있으면 그대로, 없으면 자동값
        vmax = vmax if vmax is not None else vmax2

    plt.imshow(arr, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title(title)                                # 제목 추가
    plt.colorbar()                                  # 컬러바 추가
    plt.tight_layout()                              # 레이아웃 자동 정리
    plt.savefig(path)                               # 파일 저장
    plt.close()                                     # Figure 닫기 (메모리 절약)

    print(f"[SAVE] {path.resolve()}")               # 저장 경로 로그 출력


def save_bool_mask(png_path: Path, title: str, mask: np.ndarray):
    """bool 마스크를 0/1 히트맵으로 저장"""
    arr = np.where(mask, 1.0, 0.0).astype(np.float32)
    save_map(png_path, title, arr, vmin=0.0, vmax=1.0)



def save_sequence_analysis_maps(
    *,
    seq: str,
    out_dir: Path,
    state,
    mean_v: np.ndarray,
    std_v: np.ndarray,
    unique_cnt: np.ndarray,
    static_change_rate: np.ndarray,
    speed_min_samples: int,
    speed_vmax: Optional[float],
    std_vmax: Optional[float],
    v_slow: float,
    v_fast: float,
    std_high: float,
    dwell_high: int,
    uid_low: int,
    sc_low: float,
    sc_high: float,
    slow_min: float,
    slow_max: float,
):
    reliable = (state.cnt_v >= speed_min_samples)

    std_masked = np.where(reliable, std_v, np.nan)
    save_map(
        out_dir / "std_speed_masked.png",
        f"{seq} std speed (n>={speed_min_samples})",
        std_masked,
        vmin=0.0,
        vmax=speed_vmax,
    )

    stopngo = np.where(reliable & (mean_v <= v_slow), std_v, np.nan)
    save_map(
        out_dir / "stopngo.png",
        f"{seq} stop&go (std where mean<={v_slow} m/s)",
        stopngo,
        vmin=0.0,
        vmax=std_vmax,
    )

    save_map(out_dir / "speed_samples.png", f"{seq} speed samples per cell", state.cnt_v.astype(float))
    save_map(out_dir / "unique_ids.png", f"{seq} unique track IDs", unique_cnt.astype(float))
    save_map(out_dir / "dwell.png", f"{seq} dwell (frames)", state.dwell.astype(float))
    save_map(out_dir / "mean_speed.png", f"{seq} mean speed (m/s)", mean_v, vmin=0.0, vmax=speed_vmax)
    save_map(out_dir / "std_speed.png", f"{seq} std speed (m/s)", std_v, vmin=0.0, vmax=std_vmax)

    save_map(
        out_dir / "static_dwell.png",
        f"{seq} static dwell (frames)",
        state.static_dwell.astype(float),
    )
    save_map(
        out_dir / "static_change_rate.png",
        f"{seq} static change rate",
        static_change_rate,
        vmin=0.0,
        vmax=None,
    )

    ego_stop_mask = reliable & (mean_v <= v_slow) & (static_change_rate <= sc_low) & (state.static_dwell >= 1)
    congestion_mask = reliable & (mean_v <= v_slow) & (state.dwell >= dwell_high) & (unique_cnt <= uid_low)
    stopngo_mask = reliable & (mean_v <= v_slow) & (std_v >= std_high)
    freeflow_mask = reliable & (mean_v >= v_fast) & (std_v < std_high) & (unique_cnt > uid_low) & (static_change_rate >= sc_high)
    slowmoving_mask = reliable & (mean_v > slow_min) & (mean_v <= slow_max) & (static_change_rate <= sc_high)

    save_bool_mask(out_dir / "ego_stop_mask.png", f"{seq} ego-stop mask", ego_stop_mask)
    save_bool_mask(out_dir / "congestion_mask.png", f"{seq} congestion mask", congestion_mask)
    save_bool_mask(out_dir / "stopngo_mask.png", f"{seq} stop&go mask", stopngo_mask)
    save_bool_mask(out_dir / "freeflow_mask.png", f"{seq} freeflow mask", freeflow_mask)
    save_bool_mask(out_dir / "slowmoving_mask.png", f"{seq} slow-moving mask", slowmoving_mask)

    final_cls = np.full(mean_v.shape, 255, dtype=np.uint8)
    final_cls[reliable] = 0
    final_cls[slowmoving_mask] = 4
    final_cls[stopngo_mask] = 3
    final_cls[congestion_mask] = 2
    final_cls[ego_stop_mask] = 1

    final_vis = final_cls.astype(np.float32)
    final_vis[final_vis == 255] = np.nan

    save_map(
        out_dir / "final_classmap.png",
        f"{seq} final classmap (0=free,1=ego,2=cong,3=s&g,4=slow)",
        final_vis,
        vmin=0,
        vmax=4,
    )

    sampled_mask = (state.cnt_v > 0)
    total_cells = int(np.sum(sampled_mask))
    print(f"  [SUMMARY] cells with speed samples: {total_cells}")
    if total_cells > 0:
        print(f"           mean(speed) over sampled cells = {float(np.nanmean(mean_v[sampled_mask])):.2f} m/s")
        print(f"           median(speed) over sampled cells = {float(np.nanmedian(mean_v[sampled_mask])):.2f} m/s")
