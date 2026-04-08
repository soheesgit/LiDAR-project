# kitti_io.py
# KITTI/semanticKITTI의 .bin과 .label 파일을 읽고 서로 짝지음

from __future__ import annotations

from pathlib import Path
import re
import numpy as np

# 0xFFFF(하위 16비트만 1)로 비트 AND 연산을 하면 → 순수 semantic class ID만 추출(17 이상은 인스턴스ID)
SEM_MASK = np.uint32(0xFFFF)

# 파일 정렬
def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

# 같은 이름(stem)을 가진 LiDAR 포인트 파일(.bin)과 라벨 파일(.label)을 짝지어서 리스트로 반환
def list_pairs(velo_dir: Path, lbl_dir: Path):
    bins = sorted(velo_dir.glob("*.bin"), key=lambda p: natural_key(p.stem))
    lbs = sorted(lbl_dir.glob("*.label"), key=lambda p: natural_key(p.stem))
    m_bin = {p.stem: p for p in bins}
    m_lbl = {p.stem: p for p in lbs}
    common = sorted(set(m_bin.keys()) & set(m_lbl.keys()), key=natural_key)
    return [(m_bin[k], m_lbl[k]) for k in common]

# KITTI의 Velodyne .bin 파일을 읽어서 (N,4) 형태의 numpy 배열로 반환. 각 포인트: (x, y, z), r(강도)
def read_bin_xyzr(path: Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)   # .bin 파일을 float32 배열로 읽기
    if arr.size % 4 != 0:                       # 데이터 크기가 4의 배수가 아니면 잘못된 파일
        raise ValueError(f"Invalid .bin: {path}")
    return arr.reshape(-1, 4)                   # (N,4)로 reshape → [x,y,z,reflectance]

# SemanticKITTI의 .label 파일을 읽어서 (N,) 배열로 반환. SEM_MASK = 0xFFFF를 적용해서 semantic ID만 추출.
def read_sem_labels(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint32)   # .label 파일을 부호 없는 32비트 정수 배열로 읽기
    return (raw & SEM_MASK).astype(np.int32)   # 하위 16비트만 추출 → semantic class ID만 반환
