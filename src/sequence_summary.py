# sequence_summary.py
# 전체 시퀀스 누적 맵으로 시퀀스-level feature를 계산하고, window 로그를 최종 scorevote 결과와 연결해 저장

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional

from event_encoder import encode_event_type, EncoderConfig
from final_event_scorevote import (
    ScoreVoteConfig,
    aggregate_final_event_scorevote,
    attach_sequence_summary,
    save_final_event_scorevote,
    save_window_events_jsonl,
)


@dataclass
class SequenceSummary:
    mean_v: np.ndarray
    std_v: np.ndarray
    unique_cnt: np.ndarray
    static_change_rate: np.ndarray
    event_type: str
    feats: dict
    final_result: object
    summary_path: Path


def compute_speed_stats(sum_v: np.ndarray, sum_v2: np.ndarray, cnt_v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W = cnt_v.shape
    mean_v = np.full((H, W), np.nan, dtype=np.float32)
    std_v = np.full((H, W), np.nan, dtype=np.float32)

    m_any = cnt_v > 0
    mean_v = np.divide(
        sum_v.astype(np.float32),
        cnt_v.astype(np.float32),
        out=np.full((H, W), np.nan, dtype=np.float32),
        where=m_any,
    ).astype(np.float32)

    m_std = cnt_v >= 3
    if np.any(m_std):
        mean_v2 = np.divide(
            sum_v2.astype(np.float32),
            cnt_v.astype(np.float32),
            out=np.full((H, W), np.nan, dtype=np.float32),
            where=m_std,
        )
        var = mean_v2 - (mean_v ** 2)
        var = np.where(m_std, var, np.nan)
        var = np.clip(var, 0.0, None)
        std_v = np.sqrt(var).astype(np.float32)

    return mean_v, std_v, m_any


def compute_unique_count_map(unique_sets: list[list[set[int]]], H: int, W: int) -> np.ndarray:
    return np.array(
        [[len(unique_sets[y][x]) for x in range(W)] for y in range(H)],
        dtype=np.int32,
    )


def compute_static_change_rate(static_change_count: np.ndarray, num_frames: int) -> np.ndarray:
    t_eff = max(1, num_frames - 1)
    return static_change_count.astype(np.float32) / float(t_eff)


def summarize_sequence(
    *,
    state,
    H: int,
    W: int,
    num_frames: int,
    speed_min_samples: int,
    window_logs: list[dict],
    out_dir: Path,
    enc_cfg: EncoderConfig,
    scorevote_cfg: ScoreVoteConfig,
    enc_debug: bool = False,
) -> SequenceSummary:
    mean_v, std_v, m_any = compute_speed_stats(
        state.sum_v,
        state.sum_v2,
        state.cnt_v,
    )

    unique_cnt = compute_unique_count_map(state.unique_sets, H, W)
    static_change_rate = compute_static_change_rate(state.static_change_count, num_frames)

    if enc_debug:
        focus_mask_seq = (state.dwell > 0)
        focus_n_seq = int(np.sum(focus_mask_seq))
        reliable_seq = focus_mask_seq & (state.cnt_v >= speed_min_samples)
        reliable_n_seq = int(np.sum(reliable_seq))

        if focus_n_seq > 0:
            ss_focus_seq = state.cnt_v[focus_mask_seq]
            ss_mean_seq = float(np.mean(ss_focus_seq))
            ss_min_seq = int(np.min(ss_focus_seq))
            ss_max_seq = int(np.max(ss_focus_seq))
        else:
            ss_mean_seq, ss_min_seq, ss_max_seq = float("nan"), -1, -1

        print(
            f"[DBG SEQ] focus_n={focus_n_seq} reliable_n={reliable_n_seq} "
            f"SPEED_MIN_SAMPLES={speed_min_samples} "
            f"ss_mean={ss_mean_seq:.3f} ss_min={ss_min_seq} ss_max={ss_max_seq}"
        )

    event_type, feats = encode_event_type(
        unique_cnt_map=unique_cnt.astype(np.float32),
        mean_speed_map=mean_v.astype(np.float32),
        dwell_map=state.dwell.astype(np.float32),
        cfg=enc_cfg,
        static_dwell_map=state.static_dwell.astype(np.float32),
        static_change_rate=static_change_rate.astype(np.float32),
        speed_samples_map=state.cnt_v.astype(np.float32),
    )

    save_window_events_jsonl(out_dir, window_logs)

    final_result = aggregate_final_event_scorevote(
        window_logs=window_logs,
        cfg=scorevote_cfg,
    )
    final_result = attach_sequence_summary(
        result=final_result,
        seq_event_type=event_type,
        seq_feats=feats,
    )
    summary_path = save_final_event_scorevote(
        out_dir=out_dir,
        result=final_result,
    )

    print(
        f"[FINAL] {final_result.final_event_type} "
        f"(conf={final_result.confidence:.3f}) -> {summary_path}"
    )

    return SequenceSummary(
        mean_v=mean_v,
        std_v=std_v,
        unique_cnt=unique_cnt,
        static_change_rate=static_change_rate,
        event_type=event_type,
        feats=feats,
        final_result=final_result,
        summary_path=summary_path,
    )

def save_sequence_summary_outputs(
    *,
    out_dir: Path,
    window_logs: List[Dict[str, object]],
    seq_event_type: str,
    seq_feats: Dict[str, object],
    scorevote_cfg: Optional[ScoreVoteConfig] = None,
) -> Dict[str, object]:
    scorevote_cfg = scorevote_cfg or ScoreVoteConfig()

    window_jsonl_path = save_window_events_jsonl(
        out_dir=out_dir,
        window_logs=window_logs,
    )

    final_result = aggregate_final_event_scorevote(
        window_logs=window_logs,
        cfg=scorevote_cfg,
    )

    final_result = attach_sequence_summary(
        result=final_result,
        seq_event_type=seq_event_type,
        seq_feats=seq_feats,
    )

    final_txt_path = save_final_event_scorevote(
        out_dir=out_dir,
        result=final_result,
    )

    return {
        "window_jsonl_path": window_jsonl_path,
        "final_txt_path": final_txt_path,
        "final_result": final_result,
    }
