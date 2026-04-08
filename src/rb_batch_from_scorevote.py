# rb_batch_from_scorevote.py
# 각 구간의 최종 상태 읽어 RR/PF/Ours 등 RB 스케줄러 시뮬레이션을 구간별로 돌리고 결과 CSV와 그래프 생성

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from rb_simulator import SimConfig, simulate_once

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


def read_final_event_type(scorevote_path: Path) -> str:
    with scorevote_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("final_event_type:"):
                return line.split(":", 1)[1].strip()
    raise ValueError(f"final_event_type not found in {scorevote_path}")


def parse_args():
    import argparse
    default_cfg = SimConfig()

    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        default=r"D:\연구실\new_hitmap_ver\src\out_bev_ranges"
    )
    p.add_argument("--total-rb", type=int, default=default_cfg.total_rb)
    p.add_argument("--n-vehicles", type=int, default=default_cfg.n_vehicles)
    p.add_argument("--n-slots", type=int, default=default_cfg.n_slots)
    p.add_argument("--seed", type=int, default=default_cfg.seed)
    return p.parse_args()


def save_detail_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "frame_range",
                "event_file",
                "input_state",
                "scheduler",
                "total_throughput_bps",
                "avg_delay_sec",
                "fairness",
                "congestion_avg_throughput_bps",
                "normal_avg_throughput_bps",
                "empty_avg_throughput_bps",
                "mean_active_users_per_slot",
                "mean_requested_rb_per_slot",
                "mean_allocated_rb_per_slot",
                "mean_unserved_users_per_slot",
            ]
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
        "congestion_avg_throughput_bps": 0.0,
        "normal_avg_throughput_bps": 0.0,
        "empty_avg_throughput_bps": 0.0,
        "mean_active_users_per_slot": 0.0,
        "mean_requested_rb_per_slot": 0.0,
        "mean_allocated_rb_per_slot": 0.0,
        "mean_unserved_users_per_slot": 0.0,
    })

    for row in rows:
        key = (row["input_state"], row["scheduler"])
        g = grouped[key]
        g["count"] += 1
        g["total_throughput_bps"] += float(row["total_throughput_bps"])
        g["avg_delay_sec"] += float(row["avg_delay_sec"])
        g["fairness"] += float(row["fairness"])
        g["congestion_avg_throughput_bps"] += float(row["congestion_avg_throughput_bps"])
        g["normal_avg_throughput_bps"] += float(row["normal_avg_throughput_bps"])
        g["empty_avg_throughput_bps"] += float(row["empty_avg_throughput_bps"])
        g["mean_active_users_per_slot"] += float(row["mean_active_users_per_slot"])
        g["mean_requested_rb_per_slot"] += float(row["mean_requested_rb_per_slot"])
        g["mean_allocated_rb_per_slot"] += float(row["mean_allocated_rb_per_slot"])
        g["mean_unserved_users_per_slot"] += float(row["mean_unserved_users_per_slot"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_state",
                "scheduler",
                "count",
                "mean_total_throughput_bps",
                "mean_avg_delay_sec",
                "mean_fairness",
                "mean_congestion_avg_throughput_bps",
                "mean_normal_avg_throughput_bps",
                "mean_empty_avg_throughput_bps",
                "mean_active_users_per_slot",
                "mean_requested_rb_per_slot",
                "mean_allocated_rb_per_slot",
                "mean_unserved_users_per_slot",
            ]
        )
        writer.writeheader()

        for (state, scheduler), g in sorted(grouped.items()):
            c = g["count"]
            writer.writerow({
                "input_state": state,
                "scheduler": scheduler,
                "count": c,
                "mean_total_throughput_bps": g["total_throughput_bps"] / c,
                "mean_avg_delay_sec": g["avg_delay_sec"] / c,
                "mean_fairness": g["fairness"] / c,
                "mean_congestion_avg_throughput_bps": g["congestion_avg_throughput_bps"] / c,
                "mean_normal_avg_throughput_bps": g["normal_avg_throughput_bps"] / c,
                "mean_empty_avg_throughput_bps": g["empty_avg_throughput_bps"] / c,
                "mean_active_users_per_slot": g["mean_active_users_per_slot"] / c,
                "mean_requested_rb_per_slot": g["mean_requested_rb_per_slot"] / c,
                "mean_allocated_rb_per_slot": g["mean_allocated_rb_per_slot"] / c,
                "mean_unserved_users_per_slot": g["mean_unserved_users_per_slot"] / c,
            })

    print(f"[SAVE] {out_csv}")


def save_professor_graphs(summary_csv: Path, out_dir: Path) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.read_csv(summary_csv, encoding="utf-8-sig")

    scheduler_order = ["MaxThroughput", "PF", "Ours", "OursPF", "RR"]
    state_order = ["Empty", "Normal", "Congestion"]

    if "scheduler" in df.columns:
        df["scheduler"] = pd.Categorical(
            df["scheduler"], categories=scheduler_order, ordered=True
        )
    if "input_state" in df.columns:
        df["input_state"] = pd.Categorical(
            df["input_state"], categories=state_order, ordered=True
        )

    df = df.sort_values(["input_state", "scheduler"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_grouped_bar_plot(
        *,
        data: pd.DataFrame,
        y_col: str,
        title: str,
        ylabel: str,
        filename: str,
        ylim=None,
    ) -> None:
        pivot = data.pivot(index="input_state", columns="scheduler", values=y_col)
        pivot = pivot.reindex(index=state_order, columns=scheduler_order)
        pivot = pivot.dropna(how="all")

        if pivot.empty:
            print(f"[WARN] skip empty plot: {filename}")
            return

        x = np.arange(len(pivot.index))
        width = 0.16

        plt.figure(figsize=(10, 5))
        for i, scheduler in enumerate(pivot.columns):
            vals = pivot[scheduler].values.astype(float)
            plt.bar(
                x + i * width - width * (len(pivot.columns) - 1) / 2,
                vals,
                width=width,
                label=scheduler,
            )

        plt.xticks(x, pivot.index)
        plt.xlabel("Input State")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / filename
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[SAVE] {out_path}")

    def save_single_state_bar_plot(
        *,
        state_name: str,
        y_col: str,
        title: str,
        ylabel: str,
        filename: str,
        ylim=None,
    ) -> None:
        sub = data_by_state.get(state_name)
        if sub is None or sub.empty:
            print(f"[WARN] skip empty plot: {filename}")
            return

        sub = sub.sort_values("scheduler")

        plt.figure(figsize=(8, 5))
        plt.bar(sub["scheduler"], sub[y_col])
        plt.xlabel("Scheduler")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.tight_layout()

        out_path = out_dir / filename
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[SAVE] {out_path}")

    def save_state_vehicle_throughput_plot(
        *,
        state_name: str,
        filename: str,
    ) -> None:
        sub = data_by_state.get(state_name)
        if sub is None or sub.empty:
            print(f"[WARN] skip empty plot: {filename}")
            return

        sub = sub.sort_values("scheduler")
        x = np.arange(len(sub))
        width = 0.23

        plt.figure(figsize=(9, 5))
        plt.bar(
            x - width,
            sub["mean_congestion_avg_throughput_bps"],
            width=width,
            label="Congestion vehicles",
        )
        plt.bar(
            x,
            sub["mean_normal_avg_throughput_bps"],
            width=width,
            label="Normal vehicles",
        )
        plt.bar(
            x + width,
            sub["mean_empty_avg_throughput_bps"],
            width=width,
            label="Empty vehicles",
        )

        plt.xticks(x, sub["scheduler"])
        plt.xlabel("Scheduler")
        plt.ylabel("Average throughput (bps)")
        plt.title(f"Vehicle-type Throughput in {state_name} State")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / filename
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[SAVE] {out_path}")

    def save_tradeoff_scatter(
        *,
        state_name: str,
        x_col: str,
        y_col: str,
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str,
    ) -> None:
        sub = data_by_state.get(state_name)
        if sub is None or sub.empty:
            print(f"[WARN] skip empty plot: {filename}")
            return

        sub = sub.sort_values("scheduler")

        plt.figure(figsize=(7, 5))
        for _, row in sub.iterrows():
            x = float(row[x_col])
            y = float(row[y_col])
            label = str(row["scheduler"])
            plt.scatter(x, y, s=90)
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = out_dir / filename
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[SAVE] {out_path}")

    data_by_state = {
        state: df[df["input_state"] == state].sort_values("scheduler")
        for state in state_order
    }

    # 1) 전체 상태 비교 그래프
    save_grouped_bar_plot(
        data=df,
        y_col="mean_total_throughput_bps",
        title="Total Throughput by Input State",
        ylabel="Throughput (bps)",
        filename="01_total_throughput_by_state.png",
    )
    save_grouped_bar_plot(
        data=df,
        y_col="mean_avg_delay_sec",
        title="Average Delay by Input State",
        ylabel="Delay (sec)",
        filename="02_average_delay_by_state.png",
    )
    save_grouped_bar_plot(
        data=df,
        y_col="mean_fairness",
        title="Fairness by Input State",
        ylabel="Jain Fairness",
        filename="03_fairness_by_state.png",
        ylim=(0, 1),
    )
    save_grouped_bar_plot(
        data=df,
        y_col="mean_unserved_users_per_slot",
        title="Unserved Users per Slot by Input State",
        ylabel="Unserved Users",
        filename="04_unserved_users_by_state.png",
    )

    # 2) 상태별 차량 유형 throughput 분해
    for state_name in state_order:
        save_state_vehicle_throughput_plot(
            state_name=state_name,
            filename=f"05_vehicle_type_throughput_{state_name.lower()}.png",
        )

    # 3) 상태별 단독 비교 그래프
    for state_name in state_order:
        save_single_state_bar_plot(
            state_name=state_name,
            y_col="mean_total_throughput_bps",
            title=f"Total Throughput in {state_name} State",
            ylabel="Throughput (bps)",
            filename=f"06_{state_name.lower()}_throughput.png",
        )
        save_single_state_bar_plot(
            state_name=state_name,
            y_col="mean_avg_delay_sec",
            title=f"Average Delay in {state_name} State",
            ylabel="Delay (sec)",
            filename=f"07_{state_name.lower()}_delay.png",
        )
        save_single_state_bar_plot(
            state_name=state_name,
            y_col="mean_fairness",
            title=f"Fairness in {state_name} State",
            ylabel="Jain Fairness",
            filename=f"08_{state_name.lower()}_fairness.png",
            ylim=(0, 1),
        )
        save_single_state_bar_plot(
            state_name=state_name,
            y_col="mean_unserved_users_per_slot",
            title=f"Unserved Users per Slot in {state_name} State",
            ylabel="Unserved Users",
            filename=f"09_{state_name.lower()}_unserved.png",
        )

    # 4) trade-off scatter
    for state_name in state_order:
        save_tradeoff_scatter(
            state_name=state_name,
            x_col="mean_fairness",
            y_col="mean_total_throughput_bps",
            title=f"Throughput-Fairness Tradeoff ({state_name})",
            xlabel="Jain Fairness",
            ylabel="Total Throughput (bps)",
            filename=f"10_tradeoff_throughput_fairness_{state_name.lower()}.png",
        )
        save_tradeoff_scatter(
            state_name=state_name,
            x_col="mean_avg_delay_sec",
            y_col="mean_fairness",
            title=f"Delay-Fairness Tradeoff ({state_name})",
            xlabel="Average Delay (sec)",
            ylabel="Jain Fairness",
            filename=f"11_tradeoff_delay_fairness_{state_name.lower()}.png",
        )


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    # 출력 파일
    out_detail_csv = root / "rb_simulation_results.csv"
    out_summary_csv = root / "rb_simulation_summary.csv"

    # 시뮬레이터 설정
    cfg = SimConfig(
        total_rb=args.total_rb,
        n_vehicles=args.n_vehicles,
        n_slots=args.n_slots,
        seed=args.seed,
    )

    scorevote_files = find_scorevote_files(root)
    if not scorevote_files:
        print(f"[WARN] no final_event_scorevote.txt found under: {root}")
        return

    rows: List[Dict] = []

    for file_idx, fpath in enumerate(scorevote_files):
        try:
            state = read_final_event_type(fpath)
            seq, frame_range = extract_seq_and_range(fpath, root)

            scenario = state.strip()
            schedulers = ["RR", "MaxThroughput", "PF", "Ours", "OursPF"]
            results = []

            for scheduler in schedulers:
                metrics, _debug_rows = simulate_once(
                    scheduler_name=scheduler,
                    scenario=scenario,
                    cfg=cfg,
                    run_idx=file_idx,
                )
                results.append(metrics)

            for r in results:
                rows.append({
                    "sequence": seq,
                    "frame_range": frame_range,
                    "event_file": str(fpath),
                    "input_state": state,
                    "scheduler": r.scheduler,
                    "total_throughput_bps": r.total_throughput_bps,
                    "avg_delay_sec": r.avg_delay_sec,
                    "fairness": r.fairness,
                    "congestion_avg_throughput_bps": r.congestion_avg_throughput_bps,
                    "normal_avg_throughput_bps": r.normal_avg_throughput_bps,
                    "empty_avg_throughput_bps": r.empty_avg_throughput_bps,
                    "mean_active_users_per_slot": r.mean_active_users_per_slot,
                    "mean_requested_rb_per_slot": r.mean_requested_rb_per_slot,
                    "mean_allocated_rb_per_slot": r.mean_allocated_rb_per_slot,
                    "mean_unserved_users_per_slot": r.mean_unserved_users_per_slot,
                })

            print(f"[OK] seq={seq} range={frame_range} state={state}")

        except Exception as e:
            print(f"[FAIL] {fpath}: {e}")

    save_detail_csv(rows, out_detail_csv)
    save_summary_csv(rows, out_summary_csv)

    graph_dir = root / "professor_graphs"
    save_professor_graphs(out_summary_csv, graph_dir)

    print(f"[INFO] total simulated cases = {len(rows)}")


if __name__ == "__main__":
    main()