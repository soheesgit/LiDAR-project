# heatmap_accum.py
# 전체 파이프라인 메인, 프레임들 돌며 누적/윈도우 로그/시퀀스 요약/히트맵 저장

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
from typing import List, Optional
from sklearn.cluster import DBSCAN
from dataclasses import dataclass

from config import load_config, resolve_config_path
from event_encoder import EncoderConfig
from final_event_scorevote import ScoreVoteConfig
from event_window import EventWindow
from tracking import TrackManager
from kitti_io import list_pairs
from heatmap_writer import save_sequence_analysis_maps
from sequence_summary import summarize_sequence

from frame_processing import (
    init_sequence_state,
    process_frame_step,
)

# ----------------------------------------------------------------------
# 사용자 조정 영역
# ----------------------------------------------------------------------
N_HEAD = 2000           # 각 시퀀스에서 앞에서 이 개수만 처리 (전체 돌리려면 None 또는 음수)
ASSOC_DIST = 3.0        # 트래킹 매칭 최대 거리(미터)
MAX_AGE = 3             # 이 프레임 수만큼 미검출되면 트랙 종료

# (옵션) 평균/표준편차 속도 컬러바 최대 고정: None이면 자동 범위
SPEED_VMAX: Optional[float] = 30.0   # 예: 30 m/s ≈ 108 km/h
STD_VMAX:   Optional[float] = 10.0   # 예: 10 m/s


# ----------------------------------------------------------------------
# BEV/셀 변환
# ----------------------------------------------------------------------
"""실수 좌표(x,y)를 그리드 셀 인덱스(iy,ix)로 변환"""
def xy_to_cell(x: np.ndarray, y: np.ndarray, x_min, x_max, y_min, y_max, res):
    # 주어진 x,y 좌표를 ROI 범위와 해상도(res) 기반으로 셀 단위 위치로 변환
    rx = (x - x_min) / res  # x좌표 → 그리드 상대 위치 (실수)
    ry = (y - y_min) / res  # y좌표 → 그리드 상대 위치 (실수)

    ix = np.floor(rx).astype(np.int32)  # x축 셀 인덱스 (정수로 변환)
    iy = np.floor(ry).astype(np.int32)  # y축 셀 인덱스

    H = int((y_max - y_min) / res)  # y방향 셀 개수 (격자 높이)
    W = int((x_max - x_min) / res)  # x방향 셀 개수 (격자 너비)

    # ROI 안에 있는 점만 True (격자 바깥 점은 False)
    m = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)

    return iy, ix, m, H, W

# ----------------------------------------------------------------------
# 클러스터링
# ----------------------------------------------------------------------
def cluster_vehicle_xy(xy: np.ndarray, eps: float, min_samples: int, min_pts: int) -> List[np.ndarray]:
    """
    입력: xy (N,2)   → N개의 (x,y) 좌표 점들 (차량으로 분류된 포인트들)
    출력: 각 클러스터의 (M,2) 좌표 배열 리스트
         - DBSCAN으로 클러스터링
         - 노이즈(label=-1)는 제외
         - 포인트 개수가 min_pts 미만인 작은 군집도 제외
    """
    # 포인트가 하나도 없으면 바로 빈 리스트 반환
    if xy.shape[0] == 0:
        return []

    # DBSCAN 객체 생성 (eps: 반경, min_samples: 최소 이웃 수)
    # 밀도 기반 클러스터링 알고리즘, 특정 반경(eps) 안에 이웃 포인트가 min_samples 이상 있으면 하나의 군집으로 묶음
    db = DBSCAN(eps=eps, min_samples=min_samples)

    # 각 포인트에 대해 클러스터 라벨 할당
    #   -1 → 노이즈, 0 이상 정수 → 클러스터 ID
    labels = db.fit_predict(xy)

    clusters = []
    for lab in sorted(set(labels)):  # 모든 라벨 종류 순회
        if lab == -1:
            continue  # 노이즈(-1)은 건너뜀

        # 현재 클러스터 lab에 속하는 점들만 추출
        pts = xy[labels == lab]

        # 클러스터 크기가 너무 작은 경우 (min_pts 미만) 제외
        if pts.shape[0] >= min_pts:
            clusters.append(pts)

    # 조건을 통과한 클러스터들의 점 배열 리스트 반환
    return clusters


# ----------------------------------------------------------------------
# 디버깅 유틸 (추가)
# ----------------------------------------------------------------------
def _parse_frame_spec(spec: Optional[str]) -> Optional[set]:
    """
    디버그를 특정 프레임에만 켜고 싶을 때 쓰는 파서.
    - 입력 예: "30,40,50" 또는 "30-60,100,150-170"
    - 출력: 해당 프레임 번호들의 set(int)
    """
    if spec is None:
        return None
    spec = str(spec).strip()
    if spec == "":
        return None

    frames = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a = int(a.strip())
            b = int(b.strip())
            lo, hi = (a, b) if a <= b else (b, a)
            for k in range(lo, hi + 1):
                frames.add(k)
        else:
            frames.add(int(p))
    return frames


def emit_window_log_if_ready(
    *,
    t: int,
    win: EventWindow,
    stride: int,
    win_n: int,
    speed_min_samples: int,
    enc_debug: bool,
    enc_debug_frames: Optional[set[int]],
) -> Optional[dict]:
    if (t % stride != 0) or (not win.ready):
        return None

    etype, feats = win.encode()

    if enc_debug and (enc_debug_frames is None or t in enc_debug_frames):
        try:
            focus_mask = (win.dwell > 0)            # 윈도우 내 동적 focus 후보(점유된 셀)
            focus_n = int(np.sum(focus_mask))       # focus 셀 개수

            ss = win.cnt_v.astype(np.int32)         # speed sample count
            reliable_mask = focus_mask & (ss >= speed_min_samples)  # 신뢰 셀: speed sample 충분
            reliable_n = int(np.sum(reliable_mask))     # 신뢰 셀 개수

            # focus 셀에서 speed sample 평균/최소/최대 (샘플 부족이면 여기서 평균이 낮게 나옴)
            if focus_n > 0:
                ss_focus = ss[focus_mask]
                ss_mean = float(np.mean(ss_focus))
                ss_min = int(np.min(ss_focus))
                ss_max = int(np.max(ss_focus))
            else:
                ss_mean, ss_min, ss_max = float("nan"), -1, -1

            print(
                f"[DBG WIN] t={t} focus_n={focus_n} reliable_n={reliable_n} "
                f"SPEED_MIN_SAMPLES={speed_min_samples} "
                f"ss_mean={ss_mean:.3f} ss_min={ss_min} ss_max={ss_max} "
                f"feat_speed_mean={feats.get('speed_mean', None)} "
                f"feat_speed_samples_mean={feats.get('speed_samples_mean', None)}"
            )
        except Exception as e:
            print(f"[DBG WIN] t={t} (skip debug: {type(e).__name__}: {e})")

    row = {
        "frame": int(t),
        "start": int(t - (win_n - 1)),
        "end": int(t),
        "event_type": str(etype),
    }
    row.update({k: float(v) if np.isfinite(v) else None for k, v in feats.items()})
    return row


def run_sequence(
    *,
    pairs: list,
    state,
    tracker: TrackManager,
    win: EventWindow,
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
    stride: int,
    win_n: int,
    speed_min_samples: int,
    enc_debug: bool,
    enc_debug_frames: Optional[set],
) -> list[dict]:
    window_logs = []

    for t, (bin_path, lbl_path) in enumerate(pairs):
        fr = process_frame_step(
            t=t,
            bin_path=bin_path,
            lbl_path=lbl_path,
            state=state,
            tracker=tracker,
            H=H,
            W=W,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
            res=res,
            vehicle_ids=vehicle_ids,
            static_ids=static_ids,
            eps=eps,
            min_samples=min_samples,
            min_pts=min_pts,
            chk_first=chk_first,
            chk_every=chk_every,
            xy_to_cell_fn=xy_to_cell,
            cluster_vehicle_xy_fn=cluster_vehicle_xy,
        )

        if fr is None:
            continue

        win.add(
            dwell_delta=fr.dwell_delta,
            sum_v_delta=fr.sum_v_delta,
            sum_v2_delta=fr.sum_v2_delta,
            cnt_v_delta=fr.cnt_v_delta,
            static_occ=fr.occ,
            obs=fr.obs,
            roi_count=fr.roi_count,
        )

        row = emit_window_log_if_ready(
            t=t,
            win=win,
            stride=stride,
            win_n=win_n,
            speed_min_samples=speed_min_samples,
            enc_debug=enc_debug,
            enc_debug_frames=enc_debug_frames,
        )
        if row is not None:
            window_logs.append(row)
            print("[WIN]", row)

    return window_logs


# ----------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------
def main():
    """
    전체 파이프라인 실행 엔트리포인트.
    """
    print(f"[SCRIPT] {Path(__file__).resolve()}")

    # ─────────────────────────────────────────────────────────────────
    # 0) config 로드 (+ 경로 해석)
    # ─────────────────────────────────────────────────────────────────
    p0 = argparse.ArgumentParser(add_help=False)                # 임시 파서: --config 옵션만 먼저 파싱하기 위해 생성 (도움말은 여기서 제외)
    p0.add_argument("--config", default=None)       # 실행 시 --config [파일경로] 형태로 설정 파일을 지정할 수 있음 (기본값 None)

    p0.add_argument("--seq", default=None, help="예: 07")
    p0.add_argument("--start", type=int, default=None, help="시작 프레임(예: 400)")
    p0.add_argument("--end", type=int, default=None, help="끝 프레임(예: 800, inclusive)")

    # encoder/윈도우 NaN 원인 디버깅 옵션
    p0.add_argument("--enc_debug", action="store_true", help="윈도우/시퀀스 요약 디버그 로그 출력")
    p0.add_argument("--enc_debug_frames", default=None, help="디버그 프레임 지정 (예: '30,40,50' 또는 '30-60,100')")

    p0.add_argument("--chk_first", type=int, default=10, help="초반 N프레임만 상태 체크 로그 출력")
    p0.add_argument("--chk_every", type=int, default=0, help="0이면 비활성, N이면 매 N프레임마다 체크 로그 출력")

    args0, _ = p0.parse_known_args()                            # 실제 인자에서 --config만 우선 파싱, 나머지 인자는 무시하고 남겨둠
    cfg_path = resolve_config_path(args0.config)                # 입력받은 설정 경로를 확인/보정하여 실제 존재하는 config 파일 경로로 확정

    chk_first = int(args0.chk_first)
    chk_every = int(args0.chk_every)

    # 디버그 프레임 스펙 파싱
    enc_debug = bool(args0.enc_debug)                           # 디버그 출력 여부
    enc_debug_frames = _parse_frame_spec(args0.enc_debug_frames) # 프레임 필터 (None이면 전체 프레임)

    cfg = load_config(cfg_path)         # 최종적으로 config 파일을 읽어와 cfg 객체(설정값 모음) 생성

    # ─────────────────────────────────────────────────────────────────
    # 1) 경로/ROI/클러스터 파라미터 바인딩
    # ─────────────────────────────────────────────────────────────────
    velo_root = Path(str(cfg.paths.velo_root)).resolve()
    lbl_root  = Path(str(cfg.paths.lbl_root)).resolve()
    out_root = Path(str(cfg.paths.out_root)).resolve()

    # ROI(관심 영역)와 격자 해상도: 이후 좌표→셀 인덱스 변환에 사용
    X_MIN, X_MAX = cfg.roi.x_min, cfg.roi.x_max
    Y_MIN, Y_MAX = cfg.roi.y_min, cfg.roi.y_max
    Z_MIN, Z_MAX = cfg.roi.z_min, cfg.roi.z_max
    RES = cfg.roi.resolution

    # Semantic 클래스 집합: 차량/정적
    VEHICLE_IDS = set(cfg.class_ids.vehicle)   # 차량 계열 (car, truck, bus, 기타 차량)
    STATIC_IDS  = set(cfg.class_ids.static)    # 정적 계열 (건물, 펜스, 식생, 폴/표지판 등)

    # 클러스터(DBSCAN) 파라미터
    eps         = cfg.cluster.eps                   # 한 점을 중심으로 “이웃”이라고 볼 수 있는 거리 반경
    min_samples = cfg.cluster.cluster_min_samples           # 클러스터의 “핵심 포인트(core point)”가 되기 위해 필요한 최소 이웃 수
    min_pts     = cfg.cluster.min_points_cluster    # 클러스터 전체가 유효하다고 판단하기 위한 최소 포인트 개수

    # 실험 대상 시퀀스
    seqs = list(cfg.preview.sequences)

    if args0.seq is not None:
        seqs = [args0.seq]

    # 분석/분류 임계치: config.yaml::analysis 섹션에서 가져옴
    A = cfg.analysis
    V_SLOW, V_FAST    = A.v_slow, A.v_fast          # 느림/빠름 경계 (m/s)
    STD_HIGH          = A.std_high                  # 표준편차 '높음' 기준 (m/s)
    DWELL_HIGH        = A.dwell_high                # 차량 체류 프레임 '높음' 기준
    UID_LOW           = A.uid_low                   # 고유 차량 수 '낮음' 기준
    SC_LOW, SC_HIGH   = A.sc_low, A.sc_high         # static_change_rate 낮음/높음
    SLOW_MIN, SLOW_MAX= A.slow_min, A.slow_max      # 극저속 보정 범위 (m/s)
    SPEED_MIN_SAMPLES       = A.speed_min_samples              # 속도샘플 최소(신뢰 셀)

    # 정보 로그
    print(f"[PATH] CONFIG   : {Path(cfg_path).resolve()}")
    print(f"[PATH] VELO ROOT: {velo_root} (exists={velo_root.exists()})")
    print(f"[PATH] LBL  ROOT: {lbl_root}  (exists={lbl_root.exists()})")
    print(f"[PATH] OUT  ROOT: {out_root}  (will create if not exists)")
    print(f"[INFO] ROI x[{X_MIN},{X_MAX}] y[{Y_MIN},{Y_MAX}] z[{Z_MIN},{Z_MAX}] res={RES}")
    print(f"[INFO] CLUSTER eps={eps}, min_samples={min_samples}, min_pts={min_pts}")
    dt_sec = 1.0 / float(cfg.window.fps)
    print(f"[INFO] TRACK  assoc_dist={ASSOC_DIST}m, max_age={MAX_AGE}, dt={dt_sec:.4f}s (fps={cfg.window.fps})")
    print(f"[INFO] HEAD frames per seq: {N_HEAD}")

    # (추가) 디버그 상태 로그
    if enc_debug:
        print(f"[DBG] enc_debug=ON enc_debug_frames={sorted(list(enc_debug_frames))[:20] if enc_debug_frames else None}")

    # BEV 그리드 크기 계산 (H: y방향 셀 수, W: x방향 셀 수)
    H = int((Y_MAX - Y_MIN) / RES)
    W = int((X_MAX - X_MIN) / RES)

    # ─────────────────────────────────────────────────────────────────
    # 2) 시퀀스 단위 루프
    # ─────────────────────────────────────────────────────────────────
    for seq in seqs:
        print(f"\n[SEQ] {seq}")

        # 시퀀스별 입력 디렉토리
        velo_dir = velo_root / seq / "velodyne"
        lbl_dir  = lbl_root  / seq / "labels"
        if not velo_dir.exists() or not lbl_dir.exists():
            print("  [WARN] seq path missing:", velo_dir, lbl_dir)
            continue

        # .bin ↔ .label 페어링
        pairs = list_pairs(velo_dir, lbl_dir)
        if len(pairs) == 0:
            print("  [WARN] no pairs.")
            continue

        # 우선순위: CLI 인자 > config.preview 값 > (없으면 전체)
        cfg_start = getattr(cfg.preview, "start", None)
        cfg_end = getattr(cfg.preview, "end", None)

        range_start = args0.start if args0.start is not None else cfg_start
        range_end = args0.end if args0.end is not None else cfg_end

        if range_start is not None or range_end is not None:
            N = len(pairs)
            start = 0 if range_start is None else max(0, int(range_start))
            end = (N - 1) if range_end is None else min(int(range_end), N - 1)
            # 트래킹 안정화를 위한 이전 프레임 워밍업 1장 포함
            warmup = max(0, start - 1)
            pairs = pairs[warmup:end + 1]
            print(f"  frame range: [{start}..{end}] (warmup {warmup}) -> using {len(pairs)} frames")
        else:
            # 범위 지정이 전혀 없을 때만 N_HEAD 적용 (없애고 싶으면 N_HEAD=None)
            if N_HEAD and N_HEAD > 0:
                pairs = pairs[:N_HEAD]
            print(f"  frames to process: {len(pairs)}")

        # ── 누적 버퍼(셀 단위)
        state = init_sequence_state(H, W)

        # 트랙 관리자(근접-그리디): 단순하지만 빠르고 구현 간단
        tracker = TrackManager(ASSOC_DIST, MAX_AGE, dt_sec)

        # -----------------------------
        # Sliding Window 설정 (N=30, stride=10)
        # -----------------------------
        WIN_N = 30
        STRIDE = 10

        enc_cfg_win = EncoderConfig(
            density_cap_cell=3.0,
            cell_size_m=RES,
            n_frames=WIN_N,  # 윈도우 기준
            roi_cap_vehicles=40.0,
            v_low=2.0,
            v_ok=6.0,
            occ_high=0.6,
            topk_ratio_dyn=0.05,
            topk_ratio_stat=0.05,
            min_speed_samples=SPEED_MIN_SAMPLES,
            require_speed_samples=True,
            ego_motion_low=0.10,
        )

        win = EventWindow(H=H, W=W, win_n=WIN_N, enc_cfg=enc_cfg_win, dt_sec=dt_sec)

        # ─────────────────────────────────────────────────────────────
        # 2-1) 프레임 단위 루프
        # ─────────────────────────────────────────────────────────────
        window_logs = run_sequence(
            pairs=pairs,
            state=state,
            tracker=tracker,
            win=win,
            H=H,
            W=W,
            x_min=X_MIN,
            x_max=X_MAX,
            y_min=Y_MIN,
            y_max=Y_MAX,
            z_min=Z_MIN,
            z_max=Z_MAX,
            res=RES,
            vehicle_ids=VEHICLE_IDS,
            static_ids=STATIC_IDS,
            eps=eps,
            min_samples=min_samples,
            min_pts=min_pts,
            chk_first=chk_first,
            chk_every=chk_every,
            stride=STRIDE,
            win_n=WIN_N,
            speed_min_samples=SPEED_MIN_SAMPLES,
            enc_debug=enc_debug,
            enc_debug_frames=enc_debug_frames,
        )

        # b) 출력 디렉토리
        # 범위 소스 무관하게 range_start/range_end로 폴더명 생성
        if range_start is not None or range_end is not None:
            rs = 0 if range_start is None else int(range_start)
            re = 'XXXX' if range_end is None else f"{int(range_end):04d}"
            out_dir = out_root / f"{seq}_f{rs:04d}_{re}"
        else:
            out_dir = out_root / seq

        summary = summarize_sequence(
            state=state,
            H=H,
            W=W,
            num_frames=len(pairs),
            speed_min_samples=SPEED_MIN_SAMPLES,
            window_logs=window_logs,
            out_dir=out_dir,
            enc_cfg=EncoderConfig(
                density_cap_cell=3.0,
                cell_size_m=0.8,
                n_frames=301,
                v_low=2.0,
                v_ok=6.0,
                occ_high=0.6,
                topk_ratio_dyn=0.05,
                topk_ratio_stat=0.05,
                min_speed_samples=SPEED_MIN_SAMPLES,
                require_speed_samples=True,
                ego_motion_low=0.10,
            ),
            scorevote_cfg=ScoreVoteConfig(
                v_ok=6.0,
                empty_d_th=0.03,
                a=0.50,
                b=0.30,
                c=0.20,
                force_congestion_run=2,
            ),
            enc_debug=enc_debug,
        )

        mean_v = summary.mean_v
        std_v = summary.std_v
        unique_cnt = summary.unique_cnt
        static_change_rate = summary.static_change_rate

        save_sequence_analysis_maps(
            seq=seq,
            out_dir=out_dir,
            state=state,
            mean_v=mean_v,
            std_v=std_v,
            unique_cnt=unique_cnt,
            static_change_rate=static_change_rate,
            speed_min_samples=SPEED_MIN_SAMPLES,
            v_slow=V_SLOW,
            v_fast=V_FAST,
            std_high=STD_HIGH,
            dwell_high=DWELL_HIGH,
            uid_low=UID_LOW,
            sc_low=SC_LOW,
            sc_high=SC_HIGH,
            slow_min=SLOW_MIN,
            slow_max=SLOW_MAX,
            std_vmax=STD_VMAX,
            speed_vmax=SPEED_VMAX,
        )

if __name__ == "__main__":
    main()
