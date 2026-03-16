from __future__ import annotations  # 타입 힌트에서 미래 기능(예: 클래스 자신 타입 참조)을 사용하기 위해 필요
from dataclasses import dataclass   # 데이터 저장용 클래스를 간단히 정의하기 위한 데코레이터
from pathlib import Path            # 파일/디렉토리 경로를 객체지향적으로 다루기 위한 모듈
from typing import Tuple, Optional, Union  # 타입 힌트용
import os                          # 운영체제 관련 기능 (환경변수, 파일 경로 등)
import yaml                        # YAML 형식 설정 파일을 읽기 위해 사용

# ───────────────────────────────
# [1] 프로젝트 주요 설정 데이터 구조 정의
# ───────────────────────────────

@dataclass
class Paths:            # 데이터와 출력 경로 관리
    velo_root: Path     # LiDAR 원시 데이터(velodyne) 위치
    lbl_root: Path      # 라벨 데이터 위치
    out_root: Path      # 출력 결과 저장 경로

@dataclass
class ROI:          # BEV 변환을 위한 관심영역(Region of Interest) 설정
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    resolution: float  # 격자 해상도 (m/pixel)

@dataclass
class ClassIDs:         # 객체 분류 시 사용할 라벨 ID 목록
    vehicle: list[int]  # 차량 클래스 ID
    static: list[int]   # 고정 사물(가로수, 건물, 가드레일 등) 클래스 ID

@dataclass
class Preview:
    sequences: list[str]
    n_frames: int
    start: int | None = None
    end:   int | None = None

@dataclass
class ClusterSpec:          # DBSCAN 같은 클러스터링 알고리즘의 하이퍼파라미터
    eps: float              # 인접 거리 임계값
    cluster_min_samples: int        # 최소 샘플 수
    min_points_cluster: int # 최소 클러스터 크기

@dataclass
class WindowSpec:   # 시계열 분석을 위한 윈도우 정의
    seconds: float  # 윈도우 길이(초)
    fps: float      # 프레임 레이트 (frames per second)
    @property
    def frames(self) -> int:
        # 초 단위 길이를 실제 프레임 개수로 변환
        return int(round(self.seconds * self.fps))

@dataclass
class AnalysisSpec:             # 분석에 필요한 다양한 임계치 및 파라미터
    tau_persist_frames: int     # 객체 지속 시간 기준 (프레임 단위)
    tau_flow_dx_med: float      # 흐름/이동량 기준 (m)
    diversity_K: int            # 다양성 계산 시 K값
    front_band_m: tuple[float, float]  # 차량 전방 분석 영역 (거리 범위)

    # 속도/체류 기반 임계치
    v_slow: float            # 느린 속도 경계 (m/s)
    v_fast: float            # 빠른 속도 경계 (m/s)
    std_high: float          # 속도 표준편차 높음 기준
    dwell_high: int          # 체류 프레임 수가 높은 기준
    uid_low: int             # 고유 차량 수가 낮은 기준

    # 고정 사물 변화율 기반 임계치
    sc_low: float            # 정차 신호 (static change 낮음)
    sc_high: float           # 주행 신호 (static change 높음)

    # 극저속 보정
    slow_min: float          # 매우 느린 속도 하한
    slow_max: float          # 매우 느린 속도 상한

    # 신뢰도 필터링
    speed_min_samples: int         # 최소 유효 속도 샘플 수

@dataclass
class Config:           # 전체 설정을 하나로 묶은 구조체
    paths: Paths
    roi: ROI
    class_ids: ClassIDs
    preview: Preview
    cluster: ClusterSpec
    window: WindowSpec
    analysis: AnalysisSpec

# ───────────────────────────────
# [2] config.yaml 자동 탐색 함수
# ───────────────────────────────
def resolve_config_path(cfg_path: Optional[Union[str, Path]] = None) -> Path:
    """
    config.yaml 파일의 경로를 자동으로 탐색
    - 우선순위: 명시 인자 > 환경변수 > 현재 작업 디렉토리 > src 폴더 > 프로젝트 루트
    """
    if cfg_path:
        # 사용자가 직접 지정한 경우
        p = Path(cfg_path).expanduser()
        return p if p.is_absolute() else (Path.cwd() / p)

    # 환경변수에서 탐색
    for env in ("LIDAR_CONFIG", "NEWHITMAP_CONFIG", "CONFIG_YAML"):
        val = os.environ.get(env)
        if val:
            p = Path(val).expanduser()
            if not p.is_absolute():
                p = Path.cwd() / p
            if p.exists():
                return p

    # 기본 후보 경로들
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "config.yaml",          # 현재 작업 폴더
        here.parent / "config.yaml",         # src/config.yaml
        here.parent.parent / "config.yaml",  # 프로젝트 루트/config.yaml
    ]
    for c in candidates:
        if c.exists():
            return c

    # 못 찾으면 에러 발생
    raise FileNotFoundError(
        "config.yaml not found.\n"
        f"- CWD: {Path.cwd().resolve()}\n"
        "Searched:\n  - ./config.yaml\n  - ./src/config.yaml\n  - ./config.yaml (project root)\n"
        "Fix:\n  1) 루트에서 실행하거나\n  2) --config PATH 옵션을 주거나\n  3) 환경변수 LIDAR_CONFIG 로 경로를 지정하세요."
    )

# ───────────────────────────────
# [3] config.yaml 로드 함수
# ───────────────────────────────
def load_config(cfg_path: Optional[Union[str, Path]] = None) -> Config:
    """
    YAML 파일을 읽어 Config 객체로 변환
    """
    # config.yaml 경로 확인
    cfg_path = resolve_config_path(cfg_path)

    # YAML 로드
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # paths 부분 파싱
    paths = Paths(
        velo_root=Path(raw["paths"]["velo_root"]).resolve(),
        lbl_root=Path(raw["paths"]["lbl_root"]).resolve(),
        out_root=Path(raw["paths"]["out_root"]).resolve(),
    )
    roi = ROI(**raw["roi"])
    class_ids = ClassIDs(**raw["class_ids"])
    preview = Preview(**raw["preview"])
    cluster = ClusterSpec(**raw["cluster"])
    window = WindowSpec(**raw["window"])

    # analysis 부분 파싱
    a = raw["analysis"]

    # front_band_m 검증 (리스트/튜플 길이 2여야 함)
    fb = a["front_band_m"]
    if not (isinstance(fb, (list, tuple)) and len(fb) == 2):
        raise ValueError("analysis.front_band_m must be a 2-length list/tuple of numbers")
    front_band: Tuple[float, float] = (float(fb[0]), float(fb[1]))

    # 안전한 get: 값이 없으면 기본값 사용
    v_slow      = float(a.get("v_slow", 2.0))
    v_fast      = float(a.get("v_fast", 6.0))
    std_high    = float(a.get("std_high", 1.5))
    dwell_high  = int(a.get("dwell_high", 10))
    uid_low     = int(a.get("uid_low", 3))

    sc_low      = float(a.get("sc_low", 0.05))
    sc_high     = float(a.get("sc_high", 0.15))

    slow_min    = float(a.get("slow_min", 0.2))
    slow_max    = float(a.get("slow_max", 1.0))

    min_samples = int(a.get("speed_min_samples", a.get("min_samples", 1)))

    # AnalysisSpec 객체 생성
    analysis = AnalysisSpec(
        tau_persist_frames=a["tau_persist_frames"],
        tau_flow_dx_med=a["tau_flow_dx_med"],
        diversity_K=a["diversity_K"],
        front_band_m=front_band,

        v_slow=v_slow,
        v_fast=v_fast,
        std_high=std_high,
        dwell_high=dwell_high,
        uid_low=uid_low,
        sc_low=sc_low,
        sc_high=sc_high,
        slow_min=slow_min,
        slow_max=slow_max,
        speed_min_samples=min_samples,
    )

    # 최종 Config 객체 반환
    return Config(
        paths=paths, roi=roi, class_ids=class_ids, preview=preview,
        cluster=cluster, window=window, analysis=analysis
    )
