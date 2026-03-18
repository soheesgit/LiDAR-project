from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
from typing import List, Optional
from sklearn.cluster import DBSCAN
from dataclasses import dataclass

from src.config import load_config, resolve_config_path
from src.event_encoder import EncoderConfig
from src.final_event_scorevote import ScoreVoteConfig
from src.event_window import EventWindow
from src.tracking import TrackManager
from src.kitti_io import list_pairs, read_bin_xyzr, read_sem_labels
from src.heatmap_writer import save_map, save_bool_mask
from src.sequence_summary import summarize_sequence

from src.frame_processing import (
    init_sequence_state,
    apply_roi_filter,
    build_static_occupancy,
    apply_static_occupancy,
    build_vehicle_deltas,
    apply_vehicle_result,
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

def bev_count(x: np.ndarray, y: np.ndarray, H: int, W: int, x_min, y_min, res):
    """(선택) 좌표 카운팅 BEV 생성 (여기선 사용 안함)"""
    bev = np.zeros((H, W), dtype=np.float32)
    iy, ix, m, _, _ = xy_to_cell(x, y, x_min, x_min + W*res, y_min, y_min + H*res, res)
    if m.any():
        np.add.at(bev, (iy[m], ix[m]), 1.0)
    return bev

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

def _dbg_print(enabled: bool, msg: str):
    """디버그 출력 on/off 스위치 (추가)"""
    if enabled:
        print(msg)


@dataclass
class FrameProcessResult:
    roi_count: int
    occ: np.ndarray
    dwell_delta: np.ndarray
    sum_v_delta: np.ndarray
    sum_v2_delta: np.ndarray
    cnt_v_delta: np.ndarray
    obs: list


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


def process_frame(
    *,
    t: int,
    bin_path: Path,
    lbl_path: Path,
    state,
    tracker: TrackManager,
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
) -> Optional[FrameProcessResult]:
    pts = read_bin_xyzr(bin_path)
    sem = read_sem_labels(lbl_path)

    if pts.shape[0] != sem.shape[0]:
        print(f"  [SKIP] size mismatch at {bin_path.stem}")
        return None

    roi_frame = apply_roi_filter(
        pts=pts,
        sem=sem,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )
    x, y, sem = roi_frame.x, roi_frame.y, roi_frame.sem

    veh_m = np.isin(sem, list(vehicle_ids))
    static_m = np.isin(sem, list(static_ids))

    do_chk = (t < chk_first) or (chk_every > 0 and (t % chk_every == 0))
    if do_chk:
        print(
            f"[CHK t={t}] roi_pts={len(x)} veh_pts={int(veh_m.sum())} static_pts={int(static_m.sum())}"
        )

    # c) 정적 occupancy 생성
    occ = build_static_occupancy(
        x=x,
        y=y,
        sem=sem,
        static_ids=static_ids,
        H=H,
        W=W,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        res=res,
        xy_to_cell_fn=xy_to_cell,
    )
    apply_static_occupancy(state, occ)

    # d) 차량 delta 생성 (항상 실행)
    veh_result = build_vehicle_deltas(
        x=x,
        y=y,
        sem=sem,
        vehicle_ids=vehicle_ids,
        H=H,
        W=W,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        res=res,
        tracker=tracker,
        cluster_vehicle_xy_fn=cluster_vehicle_xy,
        xy_to_cell_fn=xy_to_cell,
        eps=eps,
        min_samples=min_samples,
        min_pts=min_pts,
    )
    apply_vehicle_result(state, veh_result)

    dwell_delta = veh_result.dwell_delta
    sum_v_delta = veh_result.sum_v_delta
    sum_v2_delta = veh_result.sum_v2_delta
    cnt_v_delta = veh_result.cnt_v_delta
    obs = veh_result.obs
    roi_count = veh_result.roi_count
    seen_tid = veh_result.seen_tid

    if do_chk:
        print(
            f"[CHK t={t}] clusters={veh_result.cluster_count} "
            f"centers={len(veh_result.centers)} "
            f"(eps={eps}, min_samples={min_samples}, min_pts={min_pts})"
        )
        print(
            f"[CHK t={t}] tracks_alive={len(tracker.tracks)} "
            f"updated={veh_result.updated_track_count}"
        )

        nz = int(np.count_nonzero(dwell_delta))
        nz_cntv = int(np.count_nonzero(cnt_v_delta))
        print(
            f"[CHK t={t}] roi_count={roi_count} seen_tid={len(seen_tid)} "
            f"dwell_delta_nz={nz} cnt_v_delta_nz={nz_cntv}"
        )

    m_cnt = (cnt_v_delta > 0)
    n_cnt = int(np.sum(m_cnt))
    n_sum_finite = int(np.sum(np.isfinite(sum_v_delta[m_cnt]))) if n_cnt > 0 else 0
    n_sum_zero = int(np.sum(sum_v_delta[m_cnt] == 0.0)) if n_cnt > 0 else 0
    n_sum_bad = int(np.sum(~np.isfinite(sum_v_delta[m_cnt]))) if n_cnt > 0 else 0

    if n_cnt > 0 and np.isfinite(sum_v_delta[m_cnt]).any():
        approx_vals = sum_v_delta[m_cnt & (cnt_v_delta == 1)]
        approx_vals = approx_vals[np.isfinite(approx_vals)]
        if approx_vals.size > 0:
            v_lo = float(np.min(approx_vals))
            v_hi = float(np.max(approx_vals))
        else:
            v_lo = v_hi = float("nan")
    else:
        v_lo = v_hi = float("nan")

    print(
        f"[DBG][t={t}] delta cnt_cells={n_cnt}, sum_finite={n_sum_finite}, "
        f"sum_zero={n_sum_zero}, sum_bad={n_sum_bad}, approx_v_range=[{v_lo:.3f},{v_hi:.3f}]"
    )

    return FrameProcessResult(
        roi_count=roi_count,
        occ=occ,
        dwell_delta=dwell_delta,
        sum_v_delta=sum_v_delta,
        sum_v2_delta=sum_v2_delta,
        cnt_v_delta=cnt_v_delta,
        obs=obs,
    )


def save_sequence_outputs(
    *,
    out_dir: Path,
    window_logs: list,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    import json

    with (out_dir / "window_events.jsonl").open("w", encoding="utf-8") as f:
        for r in window_logs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[SAVE] {out_dir / 'window_events.jsonl'}")


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
        fr = process_frame(
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
            density_cap_cell=2.0,
            cell_size_m=RES,
            n_frames=WIN_N,  # 윈도우 기준
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
                density_cap_cell=0.3,
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
            ),
            enc_debug=enc_debug,
        )

        save_sequence_outputs(
            out_dir=out_dir,
            window_logs=window_logs,
        )

        mean_v = summary.mean_v
        std_v = summary.std_v
        unique_cnt = summary.unique_cnt
        static_change_rate = summary.static_change_rate


        # e) 기초 결과 저장(샘플/속도/정적)
        save_map(out_dir / "speed_samples.png",
                 f"{seq} speed samples per cell", state.cnt_v.astype(float))

        # 신뢰 셀 정의(분류/표시에서 사용할 마스크)
        reliable = (state.cnt_v >= SPEED_MIN_SAMPLES)

        # "신뢰 셀만" 남긴 표준편차 히트맵(디버깅용)
        std_masked = np.where(reliable, std_v, np.nan)
        save_map(out_dir / "std_speed_masked.png",
                 f"{seq} std speed (n>={SPEED_MIN_SAMPLES})", std_masked, vmin=0.0, vmax=STD_VMAX)

        # stop&go 강조: 평균이 느린 셀에서의 std
        stopngo = np.where(reliable & (mean_v <= V_SLOW), std_v, np.nan)
        save_map(out_dir / "stopngo.png",
                 f"{seq} stop&go (std where mean<={V_SLOW} m/s)", stopngo, vmin=0.0, vmax=STD_VMAX)

        # 기타 기본 히트맵 저장
        save_map(out_dir / "unique_ids.png", f"{seq} unique track IDs", unique_cnt.astype(float))
        save_map(out_dir / "dwell.png",      f"{seq} dwell (frames)",    state.dwell.astype(float))
        save_map(out_dir / "mean_speed.png", f"{seq} mean speed (m/s)",  mean_v, vmin=0.0, vmax=SPEED_VMAX)
        save_map(out_dir / "std_speed.png",  f"{seq} std speed (m/s)",   std_v,  vmin=0.0, vmax=STD_VMAX)

        # 정적 채널 저장
        save_map(out_dir / "static_dwell.png",
                 f"{seq} static dwell (frames)", state.static_dwell.astype(float))
        save_map(out_dir / "static_change_rate.png",
                 f"{seq} static change rate", static_change_rate, vmin=0.0, vmax=None)

        # f) 상태별 마스크 생성
        ego_stop_mask = reliable & (mean_v <= V_SLOW) & (static_change_rate <= SC_LOW) & (state.static_dwell >= 1)
        congestion_mask = reliable & (mean_v <= V_SLOW) & (state.dwell >= DWELL_HIGH) & (unique_cnt <= UID_LOW)
        stopngo_mask = reliable & (mean_v <= V_SLOW) & (std_v >= STD_HIGH)
        freeflow_mask = reliable & (mean_v >= V_FAST) & (std_v < STD_HIGH) \
                        & (unique_cnt > UID_LOW) & (static_change_rate >= SC_HIGH)
        slowmoving_mask = reliable & (mean_v > SLOW_MIN) & (mean_v <= SLOW_MAX) \
                          & (static_change_rate <= SC_HIGH)

        save_bool_mask(out_dir / "ego_stop_mask.png",   f"{seq} ego-stop mask",   ego_stop_mask)
        save_bool_mask(out_dir / "congestion_mask.png", f"{seq} congestion mask", congestion_mask)
        save_bool_mask(out_dir / "stopngo_mask.png",    f"{seq} stop&go mask",    stopngo_mask)
        save_bool_mask(out_dir / "freeflow_mask.png",   f"{seq} freeflow mask",   freeflow_mask)
        save_bool_mask(out_dir / "slowmoving_mask.png", f"{seq} slow-moving mask", slowmoving_mask)

        # h) 최종 클래스맵(우선순위 적용: ego-stop > congestion > stop&go > slow-moving > free)
        #    unlabeled=255 → 시각화 시 NaN 처리하여 컬러바 충돌 방지
        final_cls = np.full((H, W), 255, dtype=np.uint8)

        # 1) 기본은 free(0)로 채우되, 신뢰 셀만 free로 표기 (신뢰 아님은 unlabeled 유지)
        final_cls[reliable] = 0
        final_cls[slowmoving_mask] = 4
        final_cls[stopngo_mask] = 3
        final_cls[congestion_mask] = 2
        final_cls[ego_stop_mask] = 1

        # 2) 우선순위 낮은 것부터 덮어쓰기
        final_cls[slowmoving_mask] = 4  # free < slow-moving
        final_cls[stopngo_mask] = 3  # slow-moving < stop&go
        final_cls[congestion_mask] = 2  # stop&go < congestion
        final_cls[ego_stop_mask] = 1  # congestion < ego-stop  (최우선)

        # 시각화를 위해 255를 NaN으로
        final_vis = final_cls.astype(np.float32)
        final_vis[final_vis == 255] = np.nan
        save_map(out_dir / "final_classmap.png",
                 f"{seq} final classmap (0=free,1=ego,2=cong,3=s&g,4=slow)", final_vis, vmin=0, vmax=4)

        # i) 요약 통계
        sampled_mask = (state.cnt_v > 0)
        total_cells = int(np.sum(sampled_mask))
        print(f"  [SUMMARY] cells with speed samples: {total_cells}")
        if total_cells > 0:
            print(f"           mean(speed) over sampled cells = {float(np.nanmean(mean_v[sampled_mask])):.2f} m/s")
            print(f"           median(speed) over sampled cells = {float(np.nanmedian(mean_v[sampled_mask])):.2f} m/s")

if __name__ == "__main__":
    main()
