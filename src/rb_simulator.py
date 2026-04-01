from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# 1) 기본 설정
# =========================================================
# 교수님 방향:
# - 차량마다 혼잡 상태(예: Congestion / Normal / Empty)를 가진다고 보고
# - 상태에 따라 트래픽 도착량, 채널 gain 분포를 다르게 둔다.
# - RB(Resource Block)를 여러 방식으로 배분한 뒤
#   Throughput / Delay / Fairness를 비교한다.
# - 우리 방식(Ours)은 PF 위에 얹는 방식이 아니라,
#   상태 기반 가중치로 RB를 직접 나눠주는 방식으로 구현한다.


@dataclass
class SimConfig:
    # 무선 자원 설정
    total_rb: int = 12
    bandwidth_per_rb_hz: float = 180_000.0
    slot_sec: float = 0.1
    n_slots: int = 300

    # 간단 채널 모델용 전력/잡음
    tx_power: float = 1.0
    noise_power: float = 1e-9

    # 트래픽 모델
    packet_size_bits: int = 12_000

    # 차량 수
    n_vehicles: int = 9

    # 반복 실험 수
    n_runs: int = 30

    # 난수
    seed: int = 42

    # PF 안정화 상수
    pf_epsilon: float = 1e-6

    # Ours 가중치: 교수님 예시 철학 그대로
    weight_congestion: float = 2.0
    weight_normal: float = 1.0
    weight_empty: float = 0.5


@dataclass
class Vehicle:
    vid: int
    state: str
    queue_bits: float = 0.0
    served_bits_total: float = 0.0
    delay_sum: float = 0.0
    arrivals_total_bits: float = 0.0
    hol_delay: float = 0.0


@dataclass
class Metrics:
    scheduler: str
    scenario: str
    run_idx: int
    total_throughput_bps: float
    avg_delay_sec: float
    fairness: float
    congestion_avg_throughput_bps: float
    normal_avg_throughput_bps: float
    empty_avg_throughput_bps: float


# =========================================================
# 2) 상태 관련 유틸
# =========================================================


def normalize_state_name(state: str) -> str:
    s = state.strip().lower()
    if s in ("congestion", "jam", "trafficjam"):
        return "Congestion"
    if s in ("normal",):
        return "Normal"
    if s in ("empty", "light", "sparse"):
        return "Empty"
    raise ValueError(f"Unknown state: {state}")


# 상태별 환경 파라미터
# - arrival_pkts: 슬롯당 평균 패킷 도착 수
# - gain_mu / gain_sigma: lognormal 채널 gain 분포 파라미터
STATE_ENV: Dict[str, Dict[str, float]] = {
    "Congestion": {
        "arrival_pkts": 2.2,
        "gain_mu": -1.0,
        "gain_sigma": 0.45,
    },
    "Normal": {
        "arrival_pkts": 1.5,
        "gain_mu": -0.5,
        "gain_sigma": 0.35,
    },
    "Empty": {
        "arrival_pkts": 0.8,
        "gain_mu": -0.1,
        "gain_sigma": 0.25,
    },
}


def state_weight(state: str, cfg: SimConfig) -> float:
    state = normalize_state_name(state)
    if state == "Congestion":
        return cfg.weight_congestion
    if state == "Normal":
        return cfg.weight_normal
    return cfg.weight_empty


# =========================================================
# 3) 시나리오 생성
# =========================================================
# heatmap 결과를 아직 직접 붙이지 않은 1차 시뮬레이터이므로,
# 차량 상태 조합을 시나리오 프리셋으로 만든다.
# 나중에는 heatmap 기반 분류 결과를 그대로 states 리스트로 넣으면 된다.


def build_vehicle_states(scenario: str, n_vehicles: int) -> List[str]:
    scenario = scenario.strip().lower()

    if scenario == "balanced":
        # 혼잡/보통/한산을 비슷한 비율로 섞음
        base = ["Congestion", "Normal", "Empty"]
        states = [base[i % 3] for i in range(n_vehicles)]
        return states

    if scenario == "congestion_heavy":
        # 혼잡 차량이 많은 경우
        weights = [0.6, 0.3, 0.1]
    elif scenario == "normal_heavy":
        # 보통 차량이 많은 경우
        weights = [0.2, 0.6, 0.2]
    elif scenario == "empty_heavy":
        # 한산 차량이 많은 경우
        weights = [0.1, 0.3, 0.6]
    else:
        raise ValueError(
            "scenario must be one of: balanced, congestion_heavy, normal_heavy, empty_heavy"
        )

    states = np.random.choice(
        ["Congestion", "Normal", "Empty"],
        size=n_vehicles,
        p=weights,
    )
    return [str(x) for x in states]


# =========================================================
# 4) 채널 / 전송률
# =========================================================


def sample_channel_gains(states: List[str], rng: np.random.Generator) -> np.ndarray:
    gains = np.zeros(len(states), dtype=np.float64)
    for i, s in enumerate(states):
        env = STATE_ENV[s]
        gains[i] = rng.lognormal(env["gain_mu"], env["gain_sigma"])
    return gains



def shannon_rate_bits_per_rb(gains: np.ndarray, cfg: SimConfig) -> np.ndarray:
    snr = (cfg.tx_power * gains) / max(cfg.noise_power, 1e-12)
    spectral_eff = np.log2(1.0 + snr)
    return (cfg.bandwidth_per_rb_hz * spectral_eff * cfg.slot_sec).astype(np.float64)


# =========================================================
# 5) 트래픽 도착 / 큐 처리
# =========================================================


def make_vehicles(states: List[str]) -> List[Vehicle]:
    return [Vehicle(vid=i, state=s) for i, s in enumerate(states)]



def update_arrivals(vehicles: List[Vehicle], cfg: SimConfig, rng: np.random.Generator) -> None:
    for v in vehicles:
        lam = STATE_ENV[v.state]["arrival_pkts"]
        arrivals = rng.poisson(lam=lam)
        bits = float(arrivals * cfg.packet_size_bits)

        # 큐가 비어있지 않았다면 기존 데이터는 한 슬롯 더 기다림
        if v.queue_bits > 0:
            v.hol_delay += cfg.slot_sec

        v.queue_bits += bits
        v.arrivals_total_bits += bits



def serve_queues(
    vehicles: List[Vehicle],
    alloc_rb: np.ndarray,
    rate_per_rb: np.ndarray,
) -> None:
    for i, v in enumerate(vehicles):
        capacity_bits = float(alloc_rb[i]) * float(rate_per_rb[i])
        served = min(v.queue_bits, capacity_bits)

        if served <= 0:
            continue

        v.queue_bits -= served
        v.served_bits_total += served

        # 단순 지연 근사치
        served_ratio = served / max(v.arrivals_total_bits, 1e-9)
        v.delay_sum += v.hol_delay * served_ratio * served

        # 큐가 다 비면 HOL delay 초기화
        if v.queue_bits <= 1e-9:
            v.queue_bits = 0.0
            v.hol_delay = 0.0


# =========================================================
# 6) 스케줄러
# =========================================================


def alloc_round_robin(total_rb: int, rr_cursor: int, n_users: int) -> Tuple[np.ndarray, int]:
    alloc = np.zeros(n_users, dtype=np.int32)
    for k in range(total_rb):
        alloc[(rr_cursor + k) % n_users] += 1
    new_cursor = (rr_cursor + total_rb) % n_users
    return alloc, new_cursor



def alloc_max_throughput(
    total_rb: int,
    rate_per_rb: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """
    현실형에 더 가까운 MaxThroughput:
    - RB를 한 번에 한 사용자에게 몰빵하지 않고
    - RB를 1개씩 순차적으로 할당한다.
    - 각 RB마다 가장 높은 rate_per_rb를 가진 active 차량에게 1개 부여.

    현재 단순 채널 모델에서는 RB마다 rate_per_rb가 동일하므로
    결국 높은 rate 차량이 여러 개 RB를 받을 가능성이 크지만,
    적어도 구현 관점에서는 "RB 단위 할당"이 된다.
    """
    alloc = np.zeros_like(rate_per_rb, dtype=np.int32)

    metric = rate_per_rb.copy()
    metric[~active_mask] = -1.0

    if np.all(metric < 0):
        return alloc

    for _ in range(total_rb):
        best = int(np.argmax(metric))
        if metric[best] < 0:
            break
        alloc[best] += 1

    return alloc


def alloc_proportional_fair(
    total_rb: int,
    rate_per_rb: np.ndarray,
    avg_thr: np.ndarray,
    epsilon: float,
    active_mask: np.ndarray,
) -> np.ndarray:
    """
    현실형에 더 가까운 PF:
    - RB를 1개씩 순차적으로 배정
    - 매 RB 할당마다 PF metric = rate / avg_thr 계산
    - 이미 이번 슬롯에서 RB를 받은 사용자에게는
      '가상 현재 throughput'을 반영해서 다음 RB 경쟁력이 조금씩 낮아지게 함

    이렇게 하면 PF가 한 사용자에게 12개를 무조건 몰빵하는 현상을 줄이고,
    여러 사용자에게 나눠질 수 있다.
    """
    alloc = np.zeros_like(rate_per_rb, dtype=np.int32)

    if not np.any(active_mask):
        return alloc

    # 이번 슬롯에서 받은 RB를 반영한 임시 평균 throughput
    temp_avg = avg_thr.astype(np.float64).copy()

    for _ in range(total_rb):
        metric = rate_per_rb / np.maximum(temp_avg, epsilon)
        metric[~active_mask] = -1.0

        if np.all(metric < 0):
            break

        best = int(np.argmax(metric))
        alloc[best] += 1

        # 이번 슬롯에서 RB를 하나 더 받은 효과를 반영해서
        # 다음 RB 경쟁에서는 과도한 몰빵이 줄어들도록 함
        temp_avg[best] += rate_per_rb[best] / max(1.0, total_rb)

    return alloc


def proportional_integer_allocation(total_rb: int, scores: np.ndarray) -> np.ndarray:
    """
    실수 비율 점수를 총 RB 개수에 맞게 정수 RB로 바꿔준다.
    - 먼저 비율대로 floor 할당
    - 남은 RB는 소수점이 큰 순서대로 배분
    """
    alloc = np.zeros(len(scores), dtype=np.int32)

    score_sum = float(np.sum(scores))
    if score_sum <= 0:
        return alloc

    raw = total_rb * (scores / score_sum)
    base = np.floor(raw).astype(np.int32)
    alloc += base

    remain = total_rb - int(np.sum(base))
    if remain > 0:
        frac = raw - base
        order = np.argsort(-frac)
        for idx in order[:remain]:
            alloc[idx] += 1

    return alloc



def alloc_ours_weighted_by_state(
    total_rb: int,
    vehicles: List[Vehicle],
    cfg: SimConfig,
) -> np.ndarray:
    """
    교수님 방향의 핵심 구현:
    - PF 기반이 아니라
    - 차량 상태(혼잡/보통/한산)에 따라 weight를 주고
    - 그 비율대로 RB를 직접 배분한다.

    추가로, 큐가 비어 있는 차량은 굳이 RB를 받을 필요가 적으므로
    active queue 차량만 대상으로 점수를 준다.
    """
    scores = np.zeros(len(vehicles), dtype=np.float64)

    for i, v in enumerate(vehicles):
        if v.queue_bits > 0:
            scores[i] = state_weight(v.state, cfg)
        else:
            scores[i] = 0.0

    # 모두 비활성이면 아무도 배정 안 함
    if np.sum(scores) <= 0:
        return np.zeros(len(vehicles), dtype=np.int32)

    return proportional_integer_allocation(total_rb, scores)


# =========================================================
# 7) 평가 지표
# =========================================================


def jain_fairness(x: np.ndarray) -> float:
    denom = len(x) * np.sum(x ** 2)
    if denom <= 0:
        return 0.0
    return float((np.sum(x) ** 2) / denom)



def build_metrics(
    vehicles: List[Vehicle],
    scheduler: str,
    scenario: str,
    run_idx: int,
    cfg: SimConfig,
) -> Metrics:
    total_time = cfg.n_slots * cfg.slot_sec
    user_thr = np.array([v.served_bits_total / total_time for v in vehicles], dtype=np.float64)
    fairness = jain_fairness(user_thr)

    total_served_bits = float(sum(v.served_bits_total for v in vehicles))
    total_delay = float(sum(v.delay_sum for v in vehicles))
    avg_delay = total_delay / max(total_served_bits, 1e-9)

    def mean_thr_for_state(target: str) -> float:
        vals = [v.served_bits_total / total_time for v in vehicles if v.state == target]
        return float(np.mean(vals)) if vals else 0.0

    return Metrics(
        scheduler=scheduler,
        scenario=scenario,
        run_idx=run_idx,
        total_throughput_bps=float(np.sum(user_thr)),
        avg_delay_sec=float(avg_delay),
        fairness=fairness,
        congestion_avg_throughput_bps=mean_thr_for_state("Congestion"),
        normal_avg_throughput_bps=mean_thr_for_state("Normal"),
        empty_avg_throughput_bps=mean_thr_for_state("Empty"),
    )


# =========================================================
# 8) 단일 실행 / 반복 실험
# =========================================================


def simulate_once(
    scheduler_name: str,
    scenario: str,
    cfg: SimConfig,
    run_idx: int,
) -> Metrics:
    rng = np.random.default_rng(cfg.seed + run_idx)

    # balanced는 deterministic하게 만들고,
    # heavy 계열은 run마다 랜덤 샘플링이 되도록 분리
    if scenario == "balanced":
        states = build_vehicle_states(scenario, cfg.n_vehicles)
    else:
        np.random.seed(cfg.seed + run_idx)
        states = build_vehicle_states(scenario, cfg.n_vehicles)

    vehicles = make_vehicles(states)
    n_users = len(vehicles)
    rr_cursor = 0

    avg_thr = np.full(n_users, 1.0, dtype=np.float64)

    for _slot in range(cfg.n_slots):
        update_arrivals(vehicles, cfg, rng)

        gains = sample_channel_gains([v.state for v in vehicles], rng)
        rate_per_rb = shannon_rate_bits_per_rb(gains, cfg)
        active_mask = np.array([v.queue_bits > 0 for v in vehicles], dtype=bool)

        if scheduler_name == "RR":
            alloc_rb, rr_cursor = alloc_round_robin(cfg.total_rb, rr_cursor, n_users)
            # RR도 큐가 없는 차량에 굳이 RB가 낭비되지 않게 후처리
            alloc_rb = np.where(active_mask, alloc_rb, 0)
            lost_rb = cfg.total_rb - int(np.sum(alloc_rb))
            if lost_rb > 0 and np.any(active_mask):
                active_indices = np.where(active_mask)[0]
                for k in range(lost_rb):
                    alloc_rb[active_indices[k % len(active_indices)]] += 1

        elif scheduler_name == "MaxThroughput":
            alloc_rb = alloc_max_throughput(cfg.total_rb, rate_per_rb, active_mask)

        elif scheduler_name == "PF":
            alloc_rb = alloc_proportional_fair(
                cfg.total_rb, rate_per_rb, avg_thr, cfg.pf_epsilon, active_mask
            )

        elif scheduler_name == "Ours":
            alloc_rb = alloc_ours_weighted_by_state(cfg.total_rb, vehicles, cfg)

        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        serve_queues(vehicles, alloc_rb, rate_per_rb)

        inst_thr = np.array(
            [alloc_rb[i] * rate_per_rb[i] / cfg.slot_sec for i in range(n_users)],
            dtype=np.float64,
        )
        avg_thr = 0.9 * avg_thr + 0.1 * inst_thr

    return build_metrics(vehicles, scheduler_name, scenario, run_idx, cfg)



def run_experiments(cfg: SimConfig, scenarios: List[str], schedulers: List[str]) -> List[Metrics]:
    results: List[Metrics] = []
    for scenario in scenarios:
        for scheduler in schedulers:
            for run_idx in range(cfg.n_runs):
                results.append(simulate_once(scheduler, scenario, cfg, run_idx))
    return results


# =========================================================
# 9) 결과 집계 / 저장
# =========================================================


def metrics_to_rows(metrics: List[Metrics]) -> List[Dict[str, float | str | int]]:
    rows: List[Dict[str, float | str | int]] = []
    for m in metrics:
        rows.append(
            {
                "scenario": m.scenario,
                "scheduler": m.scheduler,
                "run_idx": m.run_idx,
                "total_throughput_bps": m.total_throughput_bps,
                "avg_delay_sec": m.avg_delay_sec,
                "fairness": m.fairness,
                "congestion_avg_throughput_bps": m.congestion_avg_throughput_bps,
                "normal_avg_throughput_bps": m.normal_avg_throughput_bps,
                "empty_avg_throughput_bps": m.empty_avg_throughput_bps,
            }
        )
    return rows



def save_csv(metrics: List[Metrics], out_csv: Path) -> None:
    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = metrics_to_rows(metrics)
    if not rows:
        return

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def aggregate_results(metrics: List[Metrics]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    반환 형식:
    agg[scenario][scheduler][metric_name] = mean value
    """
    agg: Dict[str, Dict[str, Dict[str, float]]] = {}

    scenarios = sorted(set(m.scenario for m in metrics))
    schedulers = sorted(set(m.scheduler for m in metrics))

    for scenario in scenarios:
        agg[scenario] = {}
        for scheduler in schedulers:
            subset = [m for m in metrics if m.scenario == scenario and m.scheduler == scheduler]
            agg[scenario][scheduler] = {
                "total_throughput_bps": float(np.mean([x.total_throughput_bps for x in subset])),
                "avg_delay_sec": float(np.mean([x.avg_delay_sec for x in subset])),
                "fairness": float(np.mean([x.fairness for x in subset])),
                "congestion_avg_throughput_bps": float(np.mean([x.congestion_avg_throughput_bps for x in subset])),
                "normal_avg_throughput_bps": float(np.mean([x.normal_avg_throughput_bps for x in subset])),
                "empty_avg_throughput_bps": float(np.mean([x.empty_avg_throughput_bps for x in subset])),
            }

    return agg



def save_summary_txt(agg: Dict[str, Dict[str, Dict[str, float]]], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for scenario, by_scheduler in agg.items():
        lines.append(f"[Scenario] {scenario}")
        header = (
            f"{'Scheduler':<16}"
            f"{'Throughput(bps)':>18}"
            f"{'Delay(s)':>14}"
            f"{'Fairness':>12}"
            f"{'CongThr':>14}"
            f"{'NormThr':>14}"
            f"{'EmptyThr':>14}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for scheduler, vals in by_scheduler.items():
            lines.append(
                f"{scheduler:<16}"
                f"{vals['total_throughput_bps']:>18.2f}"
                f"{vals['avg_delay_sec']:>14.6f}"
                f"{vals['fairness']:>12.4f}"
                f"{vals['congestion_avg_throughput_bps']:>14.2f}"
                f"{vals['normal_avg_throughput_bps']:>14.2f}"
                f"{vals['empty_avg_throughput_bps']:>14.2f}"
            )
        lines.append("")

    out_txt.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# 10) 그래프
# =========================================================


def plot_metric_bars(
    agg: Dict[str, Dict[str, Dict[str, float]]],
    metric_name: str,
    ylabel: str,
    out_path: Path,
) -> None:
    scenarios = list(agg.keys())
    schedulers = list(next(iter(agg.values())).keys())

    x = np.arange(len(scenarios))
    width = 0.18

    plt.figure(figsize=(10, 5))
    for i, scheduler in enumerate(schedulers):
        vals = [agg[sc][scheduler][metric_name] for sc in scenarios]
        plt.bar(x + i * width - width * (len(schedulers) - 1) / 2, vals, width=width, label=scheduler)

    plt.xticks(x, scenarios)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by scenario")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()



def plot_state_throughput_bars(
    agg: Dict[str, Dict[str, Dict[str, float]]],
    target_state: str,
    out_path: Path,
) -> None:
    key_map = {
        "Congestion": "congestion_avg_throughput_bps",
        "Normal": "normal_avg_throughput_bps",
        "Empty": "empty_avg_throughput_bps",
    }
    metric_name = key_map[target_state]

    scenarios = list(agg.keys())
    schedulers = list(next(iter(agg.values())).keys())

    x = np.arange(len(scenarios))
    width = 0.18

    plt.figure(figsize=(10, 5))
    for i, scheduler in enumerate(schedulers):
        vals = [agg[sc][scheduler][metric_name] for sc in scenarios]
        plt.bar(x + i * width - width * (len(schedulers) - 1) / 2, vals, width=width, label=scheduler)

    plt.xticks(x, scenarios)
    plt.ylabel("Average throughput (bps)")
    plt.title(f"Average throughput of {target_state} vehicles")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================================================
# 11) CLI
# =========================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RB scheduler simulator (professor direction)")
    p.add_argument("--total-rb", type=int, default=12)
    p.add_argument("--n-vehicles", type=int, default=9)
    p.add_argument("--n-slots", type=int, default=300)
    p.add_argument("--n-runs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="sim_out")
    return p.parse_args()


# =========================================================
# 12) 메인 실행
# =========================================================


def main() -> None:
    args = parse_args()

    cfg = SimConfig(
        total_rb=args.total_rb,
        n_vehicles=args.n_vehicles,
        n_slots=args.n_slots,
        n_runs=args.n_runs,
        seed=args.seed,
    )

    scenarios = ["balanced", "congestion_heavy", "normal_heavy", "empty_heavy"]
    schedulers = ["RR", "MaxThroughput", "PF", "Ours"]

    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"

    print("[INFO] Start simulation")
    print(f"[INFO] total_rb={cfg.total_rb}, n_vehicles={cfg.n_vehicles}, n_slots={cfg.n_slots}, n_runs={cfg.n_runs}")

    metrics = run_experiments(cfg, scenarios, schedulers)
    agg = aggregate_results(metrics)

    save_csv(metrics, out_dir / "rb_sim_results.csv")
    save_summary_txt(agg, out_dir / "rb_sim_summary.txt")

    plot_metric_bars(agg, "total_throughput_bps", "Total throughput (bps)", plot_dir / "total_throughput.png")
    plot_metric_bars(agg, "avg_delay_sec", "Average delay (sec)", plot_dir / "average_delay.png")
    plot_metric_bars(agg, "fairness", "Jain fairness", plot_dir / "fairness.png")

    plot_state_throughput_bars(agg, "Congestion", plot_dir / "throughput_congestion_vehicles.png")
    plot_state_throughput_bars(agg, "Normal", plot_dir / "throughput_normal_vehicles.png")
    plot_state_throughput_bars(agg, "Empty", plot_dir / "throughput_empty_vehicles.png")

    print(f"[SAVE] CSV     : {out_dir / 'rb_sim_results.csv'}")
    print(f"[SAVE] SUMMARY : {out_dir / 'rb_sim_summary.txt'}")
    print(f"[SAVE] PLOTS   : {plot_dir}")


if __name__ == "__main__":
    main()
