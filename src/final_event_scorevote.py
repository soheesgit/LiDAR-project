# final_event_scorevote.py
# 여러 window의 점수를 합산해서 구간 최종 상태 결정, final_event_scorevote.txt를 저장

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import math


_LABELS = ("Empty", "Normal", "Congestion")


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@dataclass(frozen=True)
class ScoreVoteConfig:
    v_ok: float = 6.0
    empty_d_th: float = 0.03
    a: float = 0.50
    b: float = 0.30
    c: float = 0.20

    force_congestion_run: int = 2
    force_density_min: float = 0.60
    force_speed_max: float = 1.0
    force_occupancy_min: float = 0.10
    force_stopped_ratio_min: float = 0.50
    force_raw_min_count: int = 2


@dataclass
class FinalScoreVoteResult:
    final_event_type: str
    confidence: float
    weights: Dict[str, float]
    v_ok: float
    empty_d_th: float
    score_sum: Dict[str, float]
    window_count: int
    total_weight: float
    sequence_event_type: Optional[str] = None
    sequence_event_feats: Optional[Dict[str, Optional[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_window_scores (
    row: Dict[str, object],
    cfg: ScoreVoteConfig,
) -> Dict[str, float]:
    density = _clip(_safe_float(row.get("density_mean")) or 0.0)
    occupancy = _clip(_safe_float(row.get("occupancy_mean")) or 0.0)

    speed = _safe_float(row.get("speed_mean"))
    v_norm = 1.0 if speed is None else _clip(speed / max(cfg.v_ok, 1e-6))

    empty_score = (
        _clip(1.0 - (density / max(cfg.empty_d_th, 1e-6)))
        if density < cfg.empty_d_th else 0.0
    )
    congestion_score = _clip(
        cfg.a * density +
        cfg.b * (1.0 - v_norm) +
        cfg.c * occupancy
    )
    normal_score = _clip(1.0 - max(empty_score, congestion_score))

    return {
        "Empty": empty_score,
        "Normal": normal_score,
        "Congestion": congestion_score,
    }


def _max_congestion_run(window_logs: List[Dict[str, object]]) -> int:
    best = 0
    cur = 0

    for row in window_logs:
        if str(row.get("event_type", "")).strip() == "Congestion":
            cur += 1
            best = max(best, cur)
        else:
            cur = 0

    return best


def _is_raw_congestion_row(row, cfg: ScoreVoteConfig) -> bool:
    density = _safe_float(row.get("density_mean"))
    speed = _safe_float(row.get("speed_mean"))
    occupancy = _safe_float(row.get("occupancy_mean"))
    stopped_ratio = _safe_float(row.get("stopped_ratio"))

    return (
        density is not None and density >= cfg.force_density_min and
        speed is not None and speed <= cfg.force_speed_max and
        occupancy is not None and occupancy >= cfg.force_occupancy_min and
        stopped_ratio is not None and stopped_ratio >= cfg.force_stopped_ratio_min
    )


def aggregate_final_event_scorevote(
    window_logs: List[Dict[str, object]],
    cfg: Optional[ScoreVoteConfig] = None,
) -> FinalScoreVoteResult:
    cfg = cfg or ScoreVoteConfig()

    score_sum = {label: 0.0 for label in _LABELS}
    total_weight = 0.0

    for row in window_logs:
        start = int(row.get("start", 0))
        end = int(row.get("end", 0))
        weight = max(1, end - start + 1)

        partial_scores = compute_window_scores(row, cfg)

        for label, score in partial_scores.items():
            score_sum[label] += weight * score

        total_weight += weight

    window_count = len(window_logs)
    max_cong_run = _max_congestion_run(window_logs)

    # raw 기준 congestion 판별
    raw_flags = [_is_raw_congestion_row(row, cfg) for row in window_logs]

    raw_count = sum(raw_flags)

    max_raw_run = 0
    cur = 0
    for flag in raw_flags:
        if flag:
            cur += 1
            max_raw_run = max(max_raw_run, cur)
        else:
            cur = 0

    force_by_run = (max_cong_run >= cfg.force_congestion_run)
    force_by_raw = (raw_count >= cfg.force_raw_min_count)
    force_by_raw_run = (max_raw_run >= cfg.force_congestion_run)

    if window_count == 0:
        final_event = "Empty"
        confidence = 0.0
    else:
        final_event = max(score_sum.items(), key=lambda item: item[1])[0]

        # 강제 Congestion 규칙
        if force_by_run or force_by_raw or force_by_raw_run:
            final_event = "Congestion"

        denom = max(sum(score_sum.values()), 1e-9)
        confidence = score_sum.get(final_event, 0.0) / denom

    return FinalScoreVoteResult(
        final_event_type=final_event,
        confidence=float(confidence),
        weights={"a": cfg.a, "b": cfg.b, "c": cfg.c},
        v_ok=float(cfg.v_ok),
        empty_d_th=float(cfg.empty_d_th),
        score_sum={k: float(v) for k, v in score_sum.items()},
        window_count=int(window_count),
        total_weight=float(total_weight),
    )


def attach_sequence_summary(
    result: FinalScoreVoteResult,
    seq_event_type: str,
    seq_feats: Dict[str, object],
) -> FinalScoreVoteResult:
    clean_feats: Dict[str, Optional[float]] = {}
    for key, value in seq_feats.items():
        fv = _safe_float(value)
        clean_feats[key] = fv

    result.sequence_event_type = str(seq_event_type)
    result.sequence_event_feats = clean_feats
    return result


def save_final_event_scorevote(
    out_dir: Path,
    result: FinalScoreVoteResult,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "final_event_scorevote.txt"

    with path.open("w", encoding="utf-8") as f:
        f.write(f"final_event_type: {result.final_event_type}\n")
        f.write(f"confidence: {result.confidence:.3f}\n")
        f.write(
            "weights(a,b,c): "
            f"{result.weights['a']},{result.weights['b']},{result.weights['c']}\n"
        )
        f.write(f"V_OK: {result.v_ok}\n")
        f.write(f"EMPTY_D_TH: {result.empty_d_th}\n")
        f.write(f"window_count: {result.window_count}\n")
        f.write(f"total_weight: {result.total_weight}\n")
        f.write(f"score_sum: {result.score_sum}\n")

        if result.sequence_event_type is not None:
            f.write(f"sequence_event_type: {result.sequence_event_type}\n")

        if result.sequence_event_feats is not None:
            f.write(f"sequence_event_feats: {result.sequence_event_feats}\n")

    return path


def save_window_events_jsonl(
    out_dir: Path,
    window_logs: List[Dict[str, object]],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "window_events.jsonl"

    with path.open("w", encoding="utf-8") as f:
        for row in window_logs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return path
