from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class Track:
    tid: int                        # 트랙 ID (각 차량을 구분하는 고유 번호)
    center: Tuple[float, float]     # 현 프레임 중심 (m)
    n: int                          # 클러스터 포인트 수
    age: int = 0                    # "최근 몇 프레임 동안 안 보였는지" 기록하는 카운터
    history: List[Tuple[float, float]] = field(default_factory=list)  # 과거 프레임들의 중심 좌표 기록 (속도 계산에 사용)
    speed: float = 0.0              # 현 프레임 속도 (m/s)
    just_updated: bool = False      # 이번 프레임에서 실제 매칭되었는지. True → 이번 프레임에 관측된 것, False → 이번엔 관측 실패(예: 가려짐)
    has_velocity: bool = False      # 속도 값이 유효하게 계산 가능한지 여부(history에 최소 2개 이상의 좌표가 쌓여야 True)


class TrackManager:
    """
        매우 단순한 최근접-그리디 매칭 기반 트래커
        - 이번 프레임의 detection과 기존 트랙을 거리 기준으로 1:1 매칭
        - 매칭되면 center/속도 갱신, age=0, just_updated=True
        - 미매칭 트랙은 age+=1, just_updated=False
        - age>max_age면 제거
        """

    def __init__(self, assoc_dist: float, max_age: int, dt: float):
        self.tracks: Dict[int, Track] = {}      # 살아있는 트랙들 {tid: Track} 사전
        self.next_tid: int = 1                  # 다음에 만들 트랙 ID (1부터 증가)
        self.assoc_dist = assoc_dist            # 매칭 허용 최대 거리(미터). 이보다 멀면 “같은 물체”로 안 봄
        self.max_age = max_age                  # 미검출 허용 프레임 수. 이를 초과하면 트랙 제거
        self.dt = dt                            # 프레임 간 시간 간격(초). 속도 계산에 사용

    def _dist(self, p, q):
        return np.linalg.norm(np.array(p) - np.array(q))

    # 이번 프레임의 검출점(detections)을 기존 트랙들과 최근접 거리 기준으로 1:1 매칭해 각 트랙의 위치·속도·상태를 갱신,
    # 남은 검출은 새 트랙으로 생성, 오래 미검출된 트랙은 제거
    def update(self, detections: List[Tuple[float, float]], counts: List[int]) -> List[Track]:
        # 1) 모든 기존 트랙에 대해 "이번 프레임에 아직 관측되지 않음"으로 초기화하고 나이를 1 올린다.
        # - age: 몇 프레임 동안 안 보였는지 (가려짐 등)
        # - just_updated: 이번 프레임에 실제 매칭되었는지 여부
        for tr in self.tracks.values():
            tr.age += 1
            tr.just_updated = False  # 초기화

        # 2) 최근접-그리디 매칭
        unmatched_det = set(range(len(detections)))  # 아직 어떤 트랙과도 매칭되지 않은 detection 인덱스 집합
        active = list(self.tracks.items())  # self.tracks.items()의 복사본 (반복 도중 딕셔너리 변경 안전)
        for tid, tr in active:
            # best_j = 그 최소 거리를 가진 detection의 인덱스, best_d = 현재 트랙이 가장 가까운 detection까지의 최소 거리
            best_j, best_d = None, 1e9  # 해당 트랙에 대해 가장 가까운 detection 후보와 거리
            # 아직 남아있는 detection들과의 거리 중 최소를 찾는다.
            for j in list(unmatched_det):
                d = self._dist(tr.center, detections[j])  # 유클리드 거리
                if d < best_d:
                    best_d, best_j = d, j

            # 최단거리가 매칭 허용 반경(self.assoc_dist) 이하면 매칭 성립
            if best_j is not None and best_d <= self.assoc_dist:
                old = tr.center
                new = detections[best_j]

                # 위치/크기 갱신 및 이력 저장
                tr.center = new
                tr.n = counts[best_j]
                tr.history.append(new)

                # 속도 계산: '가려짐(age>0)'이면 그만큼 dt를 늘려서 나눔
                if len(tr.history) >= 2:
                    gap_frames = tr.age  # update() 맨 앞에서 age를 +1 했으므로 "연속 매칭"이면 gap_frames=1
                    effective_dt = self.dt * max(1, gap_frames)
                    tr.speed = self._dist(old, new) / effective_dt
                    tr.has_velocity = True

                # 매칭되었으니 상태 리셋/표시
                tr.age = 0
                tr.just_updated = True

                # 이 detection은 소비 처리(다른 트랙이 또 못 쓰게)
                unmatched_det.remove(best_j)

        for j in unmatched_det:
            c = detections[j]
            n = counts[j]
            tr = Track(
                tid=self.next_tid,
                center=c,
                n=n,
                age=0,
                history=[c],
                speed=0.0,
                just_updated=True,  # 이번 프레임에 관측됨
                has_velocity=False  # 이전 위치가 없어 속도 미정
            )
            self.tracks[self.next_tid] = tr
            self.next_tid += 1

        # 4) 너무 오래 관측되지 않은 트랙 제거 (유령 트랙 청소)
        #    - age > self.max_age 인 것만 골라 일괄 삭제
        dead = [tid for tid, tr in self.tracks.items() if tr.age > self.max_age]
        for tid in dead:
            del self.tracks[tid]

        # 현재 활성 트랙 리스트 반환 (상위 로직에서 just_updated/has_velocity를 활용)
        return list(self.tracks.values())
