# src/event_window.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Optional, Tuple, List

import numpy as np

# EncoderConfig: 이벤트 판정에 쓰는 각종 임계치/정규화 파라미터 모음
# encode_event: 누적된 윈도우 통계를 입력받아 (event_type, feats) 반환
from src.event_encoder import EncoderConfig, encode_event


@dataclass
class FrameDelta:
    """
    프레임 1장이 '슬라이딩 윈도우'에 기여한 값들을 묶어둔 구조체.

    왜 필요하냐?
    - 슬라이딩 윈도우는 "최근 N프레임"만 유지해야 하므로,
      새 프레임을 더할 때(+), N을 초과하면 가장 오래된 프레임을 빼야(-) 함.
    - 그래서 프레임별로 '이 프레임이 더한 양(delta)'을 저장해뒀다가,
      윈도우에서 빠질 때 그대로 누적에서 빼준다.

    각 필드는 전부 (H, W) 셀 그리드 기준
    """
    dwell: np.ndarray           # (H,W) int32 : 차량 dwell(체류) 맵의 이번 프레임 기여분
                                #               예: 어떤 셀에 차량이 있으면 그 셀에 +1
    sum_v: np.ndarray           # (H,W) float32: 속도 합 누적의 이번 프레임 기여분
                                #               (유효 속도 샘플이 있을 때만 해당 셀에 v를 더함)
    sum_v2: np.ndarray          # (H,W) float32: 속도 제곱 합 누적의 이번 프레임 기여분
                                #               std/variance 계산용
    cnt_v: np.ndarray           # (H,W) int32 : 속도 샘플 개수 누적의 이번 프레임 기여분
                                #               (유효 속도 샘플이 들어왔으면 +1)
    static_occ: np.ndarray      # (H,W) bool  : 정적(static) 점유 비트맵(이번 프레임에서 정적 물체가 점유한 셀)
    static_diff: np.ndarray     # (H,W) int32 : 정적 점유 변화(XOR) 비트맵의 이번 프레임 기여분
                                #               1이면 prev_static_occ 대비 점유 상태가 바뀐 셀

    # 이번 프레임에서 관측된 차량들(tid)의 셀좌표+속도
    obs: List[Tuple[int, int, int, float]]  # (tid, iy, ix, speed)

    # ego motion (pose 없이 정적 점유맵만으로 추정한 한 프레임 이동량 기반 속도)
    # - static_occ(prev)와 static_occ(curr)의 phase correlation으로 shift(셀 단위)를 추정
    # - shift를 meters로 바꾸고 dt로 나눠 m/s 근사
    ego_speed: float    # (m/s) 추정 ego 속도(상대)
    ego_shift_dy: int   # (cells) y방향 shift
    ego_shift_dx: int   # (cells) x방향 shift

    roi_count: int


class EventWindow:
    """
    최근 N프레임(윈도우 길이)의 통계를 계속 유지하는 슬라이딩 윈도우 누적기.

    입력(프레임 단위):
      - dwell_delta, sum_v_delta, sum_v2_delta, cnt_v_delta: 동적(차량) 채널의 프레임 기여분
      - static_occ: "정적" 채널의 이번 프레임 점유(불리언 occupancy)

    내부에서 하는 일:
      1) 프레임 기여분을 누적 맵에 더한다.
      2) 큐(deque)에 프레임 델타를 넣어둔다.
      3) 큐 길이가 N을 초과하면 oldest를 pop해서 누적에서 뺀다.
      4) encode()는 현재 누적 상태(=최근 N프레임)를 바탕으로 event_type/feats를 계산한다.

    핵심 포인트:
      - 이 클래스는 "N프레임짜리 윈도우 통계"를 항상 O(1) 갱신으로 유지하려는 목적.
        (프레임마다 전체 N을 다시 합치지 않음)
    """

    def __init__(self, H: int, W: int, win_n: int, enc_cfg: EncoderConfig, *, dt_sec: float):
        """
        H, W      : BEV 그리드 높이/너비(셀 수)
        win_n     : 슬라이딩 윈도우 길이(프레임 수). 최근 N프레임만 유지
        enc_cfg   : 이벤트 인코딩(분류/요약) 파라미터(임계치, 정규화 방식 등)
        """
        self.H, self.W = int(H), int(W)
        self.N = max(1, int(win_n))     # 윈도우가 0이 되면 안 되므로 최소 1
        self.enc_cfg = enc_cfg

        # 프레임 간 시간 간격(초) - heatmap_accum에서 cfg.window.fps로 계산해서 넘겨줌
        self.dt_sec = float(dt_sec)

        # 셀 크기(m): shift(셀) → meters 변환에 사용
        self.cell_size_m = float(getattr(enc_cfg, "cell_size_m", 0.8))

        # ------------------------------------------------------------------
        # [동적(차량) 채널] 최근 N프레임 누적 맵들
        # ------------------------------------------------------------------
        self.dwell = np.zeros((H, W), np.int32)
        # dwell: 최근 N프레임 동안 셀별 "차량이 존재했던 프레임 수" 누적(체류 시간)

        self.sum_v = np.zeros((H, W), np.float32)
        # sum_v: 셀별 속도 샘플들의 합(최근 N프레임 누적)

        self.sum_v2 = np.zeros((H, W), np.float32)
        # sum_v2: 셀별 속도 샘플 제곱합(분산/표준편차 계산용)

        self.cnt_v = np.zeros((H, W), np.int32)
        # cnt_v: 셀별 속도 샘플 개수(최근 N프레임 누적)
        #       -> mean_speed_map 만들 때 sum_v/cnt_v, std 계산 시 표본수 기준으로 사용

        # ------------------------------------------------------------------
        # [정적 채널] 최근 N프레임 누적 맵들
        # ------------------------------------------------------------------
        self.static_dwell = np.zeros((H, W), np.int32)
        # static_dwell: 최근 N프레임 동안 셀별 "정적 물체 점유 프레임 수" 누적
        #              -> 정적 패턴이 오래 유지되는 셀(가려짐/벽/구조물 등) 탐지에 사용 가능

        self.static_change = np.zeros((H, W), np.int32)
        # static_change: 최근 N프레임 동안 셀별 "정적 점유 변화 횟수" 누적
        #               -> 프레임간 XOR 변화(점유가 바뀐 횟수). ego 움직임(정지/주행) 간접 신호

        # ------------------------------------------------------------------
        # 프레임별 델타를 담아두는 큐(슬라이딩 윈도우에서 빼기 위해)
        # ------------------------------------------------------------------
        self.q: Deque[FrameDelta] = deque()

        # 이전 프레임의 정적 점유(static_occ)를 기억해두는 버퍼
        # - static_diff 계산(이번 occ vs 직전 occ XOR)을 위해 필요
        self.prev_static_occ: Optional[np.ndarray] = None  # (H,W) bool

    @property
    def ready(self) -> bool:
        """
        윈도우가 'N프레임 이상' 쌓였는지 여부.
        - event 분류를 안정적으로 하려면 윈도우가 꽉 찼을 때만 encode()를 쓰는 게 일반적.
        """
        return len(self.q) >= self.N

    # 정적 점유맵이 프레임 사이에 얼마나 (dy, dx)만큼 평행이동했는지(=ego가 움직여서 배경이 밀린 양)를 FFT 기반 phase correlation로 추정
    @staticmethod
    def _phase_corr_shift(prev_occ: np.ndarray, curr_occ: np.ndarray) -> Tuple[int, int, float]:
        """정적 점유(0/1) 그리드 두 장으로 (dy,dx) shift를 추정.
        - phase correlation(FFT 기반)으로 최대 상관 위치를 찾는다.
        - 반환: (dy, dx, peak_value)
        """

        if prev_occ is None or curr_occ is None:
            return 0, 0, 0.0

        a = prev_occ.astype(np.float32)     # 이전 프레임 정적 점유 맵 (H,W) bool
        b = curr_occ.astype(np.float32)     # 현재 프레임 정적 점유 맵 (H,W) bool

        # 점유가 너무 적으면 신뢰 불가 → shift 0
        if a.sum() < 10 or b.sum() < 10:
            return 0, 0, 0.0

        # FFT
        Fa = np.fft.fft2(a)
        Fb = np.fft.fft2(b)

        # cross-power spectrum
        R = Fa * np.conj(Fb)
        denom = np.abs(R)
        R = R / (denom + 1e-9)

        r = np.fft.ifft2(R)
        r = np.abs(r)

        # peak 위치
        iy, ix = np.unravel_index(np.argmax(r), r.shape)
        peak = float(r[iy, ix])

        H, W = r.shape
        # wrap-around 보정: peak가 절반 넘어가면 음수 shift로 해석
        dy = int(iy) if iy <= H // 2 else int(iy - H)
        dx = int(ix) if ix <= W // 2 else int(ix - W)

        return dy, dx, peak

    def add(
        self,
        *,
        dwell_delta: np.ndarray,        # 이번 프레임에서 '차량 체류'가 있었던 셀 +1 같은 맵
        sum_v_delta: np.ndarray,        # 이번 프레임에서 유효 속도 v가 들어온 셀에 v를 더한 맵
        sum_v2_delta: np.ndarray,       # 위와 동일하지만 v^2
        cnt_v_delta: np.ndarray,        # 속도 샘플 들어온 셀에 +1 한 맵
        static_occ: np.ndarray,         # 이번 프레임에서 정적 물체가 점유한 셀의 bool 맵
        obs: List[Tuple[int, int, int, float]],
        roi_count: int,
    ) -> None:
        """
        프레임 1장의 기여분(delta)을 현재 슬라이딩 윈도우 누적에 반영한다.

        내부 수행:
          1) prev_static_occ와 비교해서 static_diff(XOR)를 만든다.
          2) FrameDelta로 묶어 큐에 저장한다.
          3) 누적 맵들에 +한다.
          4) N을 초과하면 가장 오래된 델타를 pop해서 누적에서 -한다.
        """

        # ------------------------------------------------------------------
        # 1) 정적 점유 변화(static_diff) + ego shift 추정
        # ------------------------------------------------------------------
        prev = self.prev_static_occ     # (H,W) bool or None
        curr = static_occ               # (H,W) bool

        if prev is None:
            # 첫 프레임: 비교 대상이 없으니 변화/shift = 0
            static_diff = np.zeros_like(curr, dtype=np.int32)
            dy, dx, peak = 0, 0, 0.0
        else:
            # 프레임 간 점유 변화(XOR)
            static_diff = np.logical_xor(curr, prev).astype(np.int32)

            # ego motion(배경 shift) 추정: prev -> curr
            dy, dx, peak = self._phase_corr_shift(prev, curr)

        # shift(셀) → meters → speed
        ego_speed = (np.hypot(dy, dx) * self.cell_size_m) / max(self.dt_sec, 1e-6)

        # 마지막에 딱 1번만 prev 갱신
        self.prev_static_occ = curr.copy()

        # ------------------------------------------------------------------
        # 2) 이번 프레임 델타를 FrameDelta로 패키징
        # ------------------------------------------------------------------
        delta = FrameDelta(
            dwell=dwell_delta.astype(np.int32, copy=False),
            sum_v=sum_v_delta.astype(np.float32, copy=False),
            sum_v2=sum_v2_delta.astype(np.float32, copy=False),
            cnt_v=cnt_v_delta.astype(np.int32, copy=False),
            static_occ=static_occ.astype(bool, copy=False),
            static_diff=static_diff,
            obs=obs,
            ego_speed=float(ego_speed),
            ego_shift_dy=int(dy),
            ego_shift_dx=int(dx),
            roi_count=int(roi_count),
        )

        # ------------------------------------------------------------------
        # 3) 누적(+)
        # ------------------------------------------------------------------
        # 동적 채널 누적
        self.dwell += delta.dwell
        self.sum_v += delta.sum_v
        self.sum_v2 += delta.sum_v2
        self.cnt_v += delta.cnt_v

        # 정적 채널 누적
        self.static_dwell += delta.static_occ.astype(np.int32)
        self.static_change += delta.static_diff

        # 큐에 저장(나중에 윈도우에서 빠질 때 그대로 빼기 위해)
        self.q.append(delta)

        # ------------------------------------------------------------------
        # 4) 윈도우 길이 초과 시 oldest pop → 누적(-)
        # ------------------------------------------------------------------
        if len(self.q) > self.N:
            old = self.q.popleft()

            # 동적 채널에서 old 프레임의 기여분을 제거
            self.dwell -= old.dwell
            self.sum_v -= old.sum_v
            self.sum_v2 -= old.sum_v2
            self.cnt_v -= old.cnt_v

            # 정적 채널에서도 제거
            self.static_dwell -= old.static_occ.astype(np.int32)
            self.static_change -= old.static_diff

            # ------------------------------------------------------------------
            # 5) 안전 클램프(음수 방지)
            # ------------------------------------------------------------------
            # 누적/차감 과정에서 수치적/입력 오류로 음수가 될 수 있으니 0으로 클램프
            np.maximum(self.dwell, 0, out=self.dwell)
            np.maximum(self.cnt_v, 0, out=self.cnt_v)
            np.maximum(self.static_dwell, 0, out=self.static_dwell)
            np.maximum(self.static_change, 0, out=self.static_change)

    def _compute_occupancy_from_obs(self, v_stop: float, tol_cell: int) -> Tuple[float, float]:
        """
        방법 C(+속도조건):
        - 같은 tid가 '주변 tol_cell' 이내에 머물고
        - 속도가 v_stop 이하인 상태가
        - 최근 N프레임 동안 연속으로 얼마나 길었는지(run length)를 계산
        반환: (occupancy_mean, occupancy_max)  둘 다 0~1 스케일 (run_len / N)
        """
        prev_cell: Dict[int, Tuple[int, int]] = {}
        run_len: Dict[int, int] = {}

        for fr in self.q:  # 시간순
            seen = set()
            for tid, iy, ix, spd in fr.obs:
                if tid in seen:
                    continue
                seen.add(tid)

                # 속도 조건: 저속/정지일 때만 "체류"로 인정
                if not np.isfinite(spd) or spd > v_stop:
                    prev_cell[tid] = (iy, ix)
                    run_len[tid] = 0
                    continue

                prev = prev_cell.get(tid)
                if prev is None:
                    run = 1
                else:
                    dy = abs(iy - prev[0])
                    dx = abs(ix - prev[1])
                    if dy <= tol_cell and dx <= tol_cell:
                        run = run_len.get(tid, 0) + 1
                    else:
                        run = 1

                prev_cell[tid] = (iy, ix)
                run_len[tid] = run

        if not run_len:
            return 0.0, 0.0

        N = float(self.N)
        vals = np.array([v / N for v in run_len.values()], dtype=np.float32)
        return float(np.mean(vals)), float(np.max(vals))

    def encode(self) -> Tuple[str, Dict[str, float]]:
        """
        현재 슬라이딩 윈도우(최근 N프레임)의 누적 통계를 바탕으로
        event_type / feats를 계산해 반환한다.

        이 함수가 하는 일:
          A) cnt_v/sum_v/sum_v2로 mean_speed_map, std_speed_map 생성
          B) static_change_rate를 (N-1)로 나눠 0~1 근사 스케일로 정규화
          C) encode_event(...) 호출하여 최종 이벤트 분류/특징값 산출

        반환:
          (event_type: str, feats: Dict[str,float])
        """

        # ------------------------------------------------------------------
        # A) mean/std 속도맵 구성 (윈도우 누적값 기반)
        # ------------------------------------------------------------------
        # 기본은 NaN으로 채워두고, 샘플이 있는 셀만 채운다.
        mean_v = np.full((self.H, self.W), np.nan, np.float32)
        std_v = np.full((self.H, self.W), np.nan, np.float32)

        # 샘플이 1개 이상 있는 셀: 평균 속도 계산 가능
        m = self.cnt_v > 0
        mean_v = np.divide(
            self.sum_v.astype(np.float32),
            self.cnt_v.astype(np.float32),
            out=np.full((self.H, self.W), np.nan, dtype=np.float32),
            where=m
        ).astype(np.float32)

        # 샘플이 충분한 셀(예: 3개 이상)에서만 표준편차 계산(불안정 방지)
        m2 = self.cnt_v >= 3
        if np.any(m2):
            mean_v2 = np.divide(
                self.sum_v2.astype(np.float32),
                self.cnt_v.astype(np.float32),
                out=np.full((self.H, self.W), np.nan, dtype=np.float32),
                where=m2
            )

            var = mean_v2 - (mean_v ** 2)
            var = np.where(m2, var, np.nan)
            var = np.clip(var, 0.0, None)

            std_v = np.sqrt(var).astype(np.float32)

        # ------------------------------------------------------------------
        # B) static_change_rate: 정적 점유 변화율 정규화
        # ------------------------------------------------------------------
        # 기존 구현이 "T-1로 나눔" 형태였다면,
        # 슬라이딩 윈도우에서는 길이가 항상 N이므로 N-1로 나눠 변화율로 만든다.
        # - N=1이면 분모가 0이 되므로 최소 1로 방어
        T_eff = max(1, self.N - 1)
        static_change_rate = self.static_change.astype(np.float32) / float(T_eff)

        # occupancy
        occ_mean, occ_max = self._compute_occupancy_from_obs(
            v_stop=getattr(self.enc_cfg, "v_stop", 1.5),        # cfg에 없으면 1.5
            tol_cell=getattr(self.enc_cfg, "tol_cell", 1),      # cfg에 없으면 1
        )

        # --- stopped_ratio: 윈도우에서 '저속/정차 차량' 비율 ---
        v_stop = float(getattr(self.enc_cfg, "v_stop", 1.5))
        slow_cnt = 0
        move_cnt = 0

        for fr in self.q:
            seen = set()
            for tid, iy, ix, spd in fr.obs:
                if tid in seen:
                    continue
                seen.add(tid)

                if not np.isfinite(spd):
                    continue  # 속도 없으면 제외(샘플 부족)

                if spd <= v_stop:
                    slow_cnt += 1
                else:
                    move_cnt += 1

        denom = slow_cnt + move_cnt
        stopped_ratio = (slow_cnt / denom) if denom > 0 else 0.0

        roi_counts = np.array([fr.roi_count for fr in self.q], dtype=np.float32)


        # ego 속도(정적 점유 shift 기반)
        ego_speed_series = np.array([fr.ego_speed for fr in self.q], dtype=np.float32)

        # ------------------------------------------------------------------
        # C) 최종 이벤트 인코딩(분류 + feature 요약)
        # ------------------------------------------------------------------

        # 윈도우 길이(N)에 따라 dwell 누적값이 커지지 않도록 "프레임 평균"으로 정규화
        # - self.dwell: 최근 N프레임 동안 셀에 차량 중심이 찍힌 횟수 (0~N)
        # - dwell_mean: 그걸 N으로 나눈 값 (0~1) = "윈도우 프레임 중 등장 비율"
        dwell_mean = self.dwell.astype(np.float32) / float(self.N)

        etype, feats = encode_event(
            dwell_map=dwell_mean,
            mean_speed_map=mean_v,
            std_speed_map=std_v,
            speed_samples_map=self.cnt_v,
            static_dwell_map=self.static_dwell,
            static_change_rate=static_change_rate,
            cfg=self.enc_cfg,
            roi_count_series=roi_counts,
            ego_speed_series=ego_speed_series,
            stopped_ratio=float(stopped_ratio),
        )

        if feats.get("empty_like", False):
            etype = "Empty"

        feats["occupancy_mean"] = occ_mean
        feats["occupancy_max"] = occ_max
        feats["stopped_ratio"] = float(stopped_ratio)
        return etype, feats
