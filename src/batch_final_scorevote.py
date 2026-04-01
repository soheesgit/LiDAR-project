from __future__ import annotations

import subprocess
from pathlib import Path
import yaml


def count_frames(velo_dir: Path) -> int:
    return len(sorted(velo_dir.glob("*.bin")))


def make_ranges(n_frames: int, window_len: int = 300, hop: int = 100):
    ranges = []
    start = 0
    while start < n_frames:
        end = start + window_len
        if end >= n_frames:
            end = n_frames - 1
            if end <= start:
                break
            ranges.append((start, end))
            break
        ranges.append((start, end))
        start += hop
    return ranges


def write_temp_config(base_cfg_path: Path, temp_cfg_path: Path, out_root: Path):
    with base_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"]["out_root"] = str(out_root)

    temp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def main():
    project_root = Path(r"D:\연구실\new_hitmap_ver")
    py_exe = project_root / ".venv" / "Scripts" / "python.exe"
    script = project_root / "src" / "heatmap_accum.py"

    base_cfg = project_root / "src" / "config.yaml"
    velo_root = Path(r"D:\dataset\sequences")
    out_root = project_root / "src" / "out_bev_ranges"
    temp_cfg_dir = project_root / "src" / "_temp_batch_configs"

    seqs = [f"{i:02d}" for i in range(11)]   # 00 ~ 10
    window_len = 300
    hop = 100

    for seq in seqs:
        velo_dir = velo_root / seq / "velodyne"
        if not velo_dir.exists():
            print(f"[WARN] missing: {velo_dir}")
            continue

        n_frames = count_frames(velo_dir)
        ranges = make_ranges(n_frames, window_len=window_len, hop=hop)

        print(f"\n[SEQ] {seq} total_frames={n_frames} n_ranges={len(ranges)}")

        for start, end in ranges:
            run_name = f"{start:06d}_{end:06d}"
            run_out_dir = out_root / seq / run_name
            temp_cfg_path = temp_cfg_dir / f"config_{seq}_{run_name}.yaml"

            write_temp_config(
                base_cfg_path=base_cfg,
                temp_cfg_path=temp_cfg_path,
                out_root=run_out_dir,
            )

            cmd = [
                str(py_exe),
                str(script),
                "--config", str(temp_cfg_path),
                "--seq", seq,
                "--start", str(start),
                "--end", str(end),
            ]

            print(f"[RUN] seq={seq} range=[{start}..{end}]")
            result = subprocess.run(cmd, cwd=str(project_root))

            if result.returncode != 0:
                print(f"[FAIL] seq={seq} range=[{start}..{end}] code={result.returncode}")
            else:
                print(f"[OK]   seq={seq} range=[{start}..{end}] -> {run_out_dir}")


if __name__ == "__main__":
    main()