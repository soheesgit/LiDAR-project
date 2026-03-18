from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def auto_range(arr: np.ndarray):
    """NaN/inf 제외한 자동 범위 계산"""
    finite = np.isfinite(arr)           # 유한한 값(True) / NaN, inf(False) 마스크
    if not finite.any():                # 유효한 값이 하나도 없으면
        return None, None               # (None, None) 반환

    lo = float(np.nanmin(arr[finite]))  # 배열 내 최소값
    hi = float(np.nanmax(arr[finite]))  # 배열 내 최대값

    if lo == hi:                        # min=max라서 범위가 0이면
        hi = lo + 1e-6                  # hi를 살짝 늘려서 (시각화 에러 방지)

    return lo, hi                       # (최솟값, 최댓값) 반환


def save_map(path: Path, title: str, arr: np.ndarray, vmin=None, vmax=None):
    """히트맵 저장 (vmin/vmax 지정 가능)"""
    path.parent.mkdir(parents=True, exist_ok=True)      # 저장 경로 없으면 생성

    plt.figure()                                        # 새 Figure 시작
    if vmin is None or vmax is None:                    # min/max 지정 안 했을 경우
        vmin2, vmax2 = auto_range(arr)                  # auto_range로 범위 계산
        vmin = vmin if vmin is not None else vmin2      # 지정값이 있으면 그대로, 없으면 자동값
        vmax = vmax if vmax is not None else vmax2

    plt.imshow(arr, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title(title)                                # 제목 추가
    plt.colorbar()                                  # 컬러바 추가
    plt.tight_layout()                              # 레이아웃 자동 정리
    plt.savefig(path)                               # 파일 저장
    plt.close()                                     # Figure 닫기 (메모리 절약)

    print(f"[SAVE] {path.resolve()}")               # 저장 경로 로그 출력


def save_bool_mask(png_path: Path, title: str, mask: np.ndarray):
    """bool 마스크를 0/1 히트맵으로 저장"""
    arr = np.where(mask, 1.0, 0.0).astype(np.float32)
    save_map(png_path, title, arr, vmin=0.0, vmax=1.0)
