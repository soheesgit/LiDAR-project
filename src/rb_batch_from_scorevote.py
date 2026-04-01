from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from rb_simulator import SimConfig, simulate_all, read_final_event_type


def find_scorevote_files(root: Path) -> List[Path]:
    return sorted(root.rglob("final_event_scorevote.txt"))


def extract_seq_and_range(scorevote_path: Path, root: Path) -> Tuple[str, str]:
    """
    예:
    root = .../out_bev_ranges
    path = .../out_bev_ranges/00/000000_000300/00_f0000_0300/final_event_scorevote.txt

    -> seq = 00
    -> frame_range = 000000_000300
    """
    rel = scorevote_path.relative_to(root)
    parts = rel.parts

    seq = parts[0] if len(parts) >= 1 else ""
    frame_range = parts[1] if len(parts) >= 2 else ""

    return seq, frame_range


def save_detail_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "frame_range",
                "event_file",
                "state",
                "scheduler",
                "total_throughput_bps",
                "avg_delay_sec",
                "fairness",
                "ego_throughput_bps",
                "ego_avg_delay_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SAVE] {out_csv}")


def save_summary_csv(rows: List[Dict], out_csv: Path) -> None:
    """
    state + scheduler 기준 평균 요약
    """
    grouped = defaultdict(lambda: {
        "count": 0,
        "total_throughput_bps": 0.0,
        "avg_delay_sec": 0.0,
        "fairness": 0.0,
        "ego_throughput_bps": 0.0,
        "ego_avg_delay_sec": 0.0,
    })

    for row in rows:
        key = (row["state"], row["scheduler"])
        g = grouped[key]
        g["count"] += 1
        g["total_throughput_bps"] += float(row["total_throughput_bps"])
        g["avg_delay_sec"] += float(row["avg_delay_sec"])
        g["fairness"] += float(row["fairness"])
        g["ego_throughput_bps"] += float(row["ego_throughput_bps"])
        g["ego_avg_delay_sec"] += float(row["ego_avg_delay_sec"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "state",
                "scheduler",
                "count",
                "mean_total_throughput_bps",
                "mean_avg_delay_sec",
                "mean_fairness",
                "mean_ego_throughput_bps",
                "mean_ego_avg_delay_sec",
            ],
        )
        writer.writeheader()

        for (state, scheduler), g in sorted(grouped.items()):
            c = g["count"]
            writer.writerow({
                "state": state,
                "scheduler": scheduler,
                "count": c,
                "mean_total_throughput_bps": g["total_throughput_bps"] / c,
                "mean_avg_delay_sec": g["avg_delay_sec"] / c,
                "mean_fairness": g["fairness"] / c,
                "mean_ego_throughput_bps": g["ego_throughput_bps"] / c,
                "mean_ego_avg_delay_sec": g["ego_avg_delay_sec"] / c,
            })

    print(f"[SAVE] {out_csv}")


def main() -> None:
    # 네 현재 range 결과 루트
    root = Path(r"D:\연구실\new_hitmap_ver\src\out_bev_ranges")

    # 출력 파일
    out_detail_csv = root / "rb_simulation_results.csv"
    out_summary_csv = root / "rb_simulation_summary.csv"

    # 시뮬레이터 설정
    cfg = SimConfig(
        total_rb=12,
        n_bg=8,
        n_slots=300,
        seed=42,
    )

    scorevote_files = find_scorevote_files(root)
    if not scorevote_files:
        print(f"[WARN] no final_event_scorevote.txt found under: {root}")
        return

    rows: List[Dict] = []

    for fpath in scorevote_files:
        try:
            state = read_final_event_type(fpath)
            seq, frame_range = extract_seq_and_range(fpath, root)

            results = simulate_all(state, cfg)

            for r in results:
                rows.append({
                    "sequence": seq,
                    "frame_range": frame_range,
                    "event_file": str(fpath),
                    "state": r.state,
                    "scheduler": r.scheduler,
                    "total_throughput_bps": r.total_throughput_bps,
                    "avg_delay_sec": r.avg_delay_sec,
                    "fairness": r.fairness,
                    "ego_throughput_bps": r.ego_throughput_bps,
                    "ego_avg_delay_sec": r.ego_avg_delay_sec,
                })

            print(f"[OK] seq={seq} range={frame_range} state={state}")

        except Exception as e:
            print(f"[FAIL] {fpath}: {e}")

    save_detail_csv(rows, out_detail_csv)
    save_summary_csv(rows, out_summary_csv)
    print(f"[INFO] total simulated cases = {len(rows)}")


if __name__ == "__main__":
    main()