from __future__ import annotations

from typing import Optional
import numpy as np


def _safe_nanmean(x: Optional[np.ndarray]) -> float:
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


def _density_from_sumcount(sum_count_map: np.ndarray, cfg) -> np.ndarray:
    sc = sum_count_map.astype(np.float32)
    n_frames = max(1, int(cfg.n_frames))
    mean_count_cell = sc / float(n_frames)
    return np.clip(mean_count_cell / float(cfg.density_cap_cell), 0.0, 1.0)


def _density_from_meancount(mean_count_map: np.ndarray, cfg) -> np.ndarray:
    mc = mean_count_map.astype(np.float32)
    return np.clip(mc / float(cfg.density_cap_cell), 0.0, 1.0)