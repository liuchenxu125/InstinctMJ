"""Convert CASBOT02 61-column CSV .data recordings to InstinctMJ AMP motions.

Coordinate and column conventions follow amp_mjlab's casbot02_data_to_npz.py.
Output is the InstinctMJ retargeted schema, not amp_mjlab's body-state schema.
Conversion runs on CPU; the motion loader computes velocities and link FK.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from instinct_mj.assets.casbot_02 import CASBOT02_PARKOUR_JOINT_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "Datasets/casbot02"
MODEL_PATH = REPO_ROOT / "src/instinct_mj/assets/resources/casbot_23dof/xml/CASBOT_02_18dof.xml"
SELECTION_NAME = "parkour_motion_without_run.yaml"

# Zero-based columns. Fixed waist, shoulder yaw, and wrist joints are omitted.
JOINT_COLUMNS = {
    **{f"leg_l{i + 1}_joint": 12 + i for i in range(6)},
    **{f"leg_r{i + 1}_joint": 18 + i for i in range(6)},
    "upper_left_1_joint": 24,
    "upper_left_2_joint": 25,
    "upper_left_4_joint": 27,
    "upper_right_1_joint": 31,
    "upper_right_2_joint": 32,
    "upper_right_4_joint": 34,
}


def load_raw_data(path: Path) -> tuple[np.ndarray, bool]:
    """Preserve the first numeric frame; accept a text header and trailing commas.

    Reject malformed/non-finite frames rather than deleting them and changing
    the motion's timing. Blank lines do not represent source frames.
    """
    rows = []
    header_skipped = False
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line_number, fields in enumerate(csv.reader(stream), 1):
            if not fields or all(not field.strip() for field in fields):
                continue
            while fields and not fields[-1].strip():
                fields.pop()
            if len(fields) < 61:
                raise ValueError(f"{path}:{line_number}: expected at least 61 columns, got {len(fields)}")
            try:
                values = [float(field) for field in fields[:61]]
            except ValueError as exc:
                # A header must be the first non-empty row and consist entirely
                # of labels; a damaged numeric first frame is still an error.
                is_header = True
                for field in fields[:61]:
                    try:
                        float(field)
                    except ValueError:
                        continue
                    is_header = False
                    break
                if not rows and not header_skipped and is_header:
                    header_skipped = True
                    continue
                raise ValueError(f"{path}:{line_number}: invalid numeric frame") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"{path}:{line_number}: non-finite frame")
            rows.append(values)
    if not rows:
        raise ValueError(f"{path}: no numeric frames")
    return np.asarray(rows, dtype=np.float64), header_skipped


def root_quat_wxyz(raw: np.ndarray, yaw_sign: int, pitch_offset: str) -> tuple[np.ndarray, float]:
    """Raw columns 3/4/5 are pitch/roll/yaw, in radians.

    Some clips encode upright pitch as pi/2, others as zero. Choose separately
    for each clip, following the reference converter; allow explicit overrides.
    """
    mean_pitch = float(raw[:, 3].mean())
    offset = {
        "none": 0.0,
        "pi/2": np.pi / 2,
        "auto": np.pi / 2 if abs(mean_pitch - np.pi / 2) < abs(mean_pitch) else 0.0,
    }[pitch_offset]
    pitch, roll, yaw = raw[:, 3] - offset, raw[:, 4], yaw_sign * raw[:, 5]
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    quat = np.column_stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    for index in range(1, len(quat)):
        if np.dot(quat[index - 1], quat[index]) < 0:
            quat[index] *= -1
    return quat.astype(np.float32), offset


def build_motion(
    raw: np.ndarray,
    *,
    input_fps: int = 1000,
    output_fps: int = 50,
    trim_start_frames: int = 0,
    keep_seconds: float | None = None,
    root_lateral_sign: int = 1,
    root_yaw_sign: int = -1,
    pitch_offset: str = "auto",
) -> tuple[dict[str, np.ndarray], dict]:
    if input_fps <= 0 or output_fps <= 0 or input_fps % output_fps:
        raise ValueError("Expected a positive integer downsample ratio (e.g. 1000 -> 50 Hz)")
    if trim_start_frames < 0 or trim_start_frames >= len(raw):
        raise ValueError("trim_start_frames must leave at least one source frame")
    if root_lateral_sign not in (-1, 1) or root_yaw_sign not in (-1, 1):
        raise ValueError("Root coordinate signs must be -1 or 1")
    raw = raw[trim_start_frames:]
    if keep_seconds is not None:
        if not np.isfinite(keep_seconds) or keep_seconds <= 0:
            raise ValueError("keep_seconds must be finite and positive")
        raw = raw[: max(1, int(round(keep_seconds * input_fps)))]
    raw = raw[:: input_fps // output_fps]
    # The task uses ten AMP history frames. A shorter clip is not useful here.
    if len(raw) < 10:
        raise ValueError(f"Motion must contain at least 10 output frames, got {len(raw)}")
    quat, offset = root_quat_wxyz(raw, root_yaw_sign, pitch_offset)
    positions = raw[:, [1, 0, 2]].copy()
    positions[:, 1] *= root_lateral_sign
    joints = raw[:, [JOINT_COLUMNS[name] for name in CASBOT02_PARKOUR_JOINT_NAMES]]
    motion = {
        "framerate": np.asarray(float(output_fps)),
        "joint_names": np.asarray(CASBOT02_PARKOUR_JOINT_NAMES),
        "joint_pos": joints.astype(np.float32),
        "base_pos_w": positions.astype(np.float32),
        "base_quat_w": quat,
    }
    report = {
        "output_frames": len(raw),
        "framerate": output_fps,
        "duration_seconds": (len(raw) - 1) / output_fps,
        "pitch_offset_rad": offset,
        "root_position_min_m": positions.min(axis=0).tolist(),
        "root_position_max_m": positions.max(axis=0).tolist(),
        "root_height_change_m": float(positions[-1, 2] - positions[0, 2]),
        "max_abs_joint_velocity_rad_s": float(np.abs(np.diff(joints, axis=0) * output_fps).max()),
    }
    return motion, report


def validate_model_motion(model: mujoco.MjModel, motion: dict[str, np.ndarray]) -> dict:
    """Check the current model's joint order and limits without clipping data."""
    ids = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    names = [model.joint(i).name for i in ids]
    if names != motion["joint_names"].tolist():
        raise ValueError(f"Motion/model joint order mismatch: {names}")
    if len(ids) != 18:
        raise ValueError(f"Expected the current 18-joint CASBOT02 model, got {len(ids)}")
    limits = model.jnt_range[ids]
    positions = motion["joint_pos"]
    excess = np.maximum(limits[:, 0] - positions.min(axis=0), positions.max(axis=0) - limits[:, 1])
    return {name: float(value) for name, value in zip(names, excess) if value > 1e-5}


def convert_directory(args: argparse.Namespace) -> dict:
    sources = sorted(args.input_dir.glob("*.data"))
    if not sources:
        raise ValueError(f"No .data files found in {args.input_dir}")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    # Validate every source before writing the training selection.
    converted = []
    for source in sources:
        raw, header_skipped = load_raw_data(source)
        motion, report = build_motion(
            raw,
            input_fps=args.input_fps,
            output_fps=args.output_fps,
            trim_start_frames=args.trim_start_frames,
            keep_seconds=args.keep_seconds,
            root_lateral_sign=args.root_lateral_sign,
            root_yaw_sign=args.root_yaw_sign,
            pitch_offset=args.pitch_offset,
        )
        violations = validate_model_motion(model, motion)
        report.update(
            source_file=source.name,
            source_frames=len(raw),
            header_skipped=header_skipped,
            output_file=f"{source.stem}_retargeted.npz",
            joint_limit_excess_rad=violations,
        )
        if violations and not args.allow_joint_limit_violations:
            raise ValueError(f"{source}: joint limits exceeded: {violations}; inspect the data before training")
        converted.append((motion, report))

    selected_files = [report["output_file"] for _, report in converted]
    original_selection = args.input_dir / SELECTION_NAME
    missing = []
    if original_selection.is_file():
        selection = yaml.safe_load(original_selection.read_text(encoding="utf-8")) or {}
        missing = [name for name in selection.get("selected_files", []) if name not in selected_files]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for motion, report in converted:
        target = args.output_dir / report["output_file"]
        np.savez_compressed(target, **motion)
        print(
            f"[converted] {report['source_file']} -> {target.name}: "
            f"{report['output_frames']} frames, {report['duration_seconds']:.2f}s, "
            f"pitch offset={report['pitch_offset_rad']:.6f}"
        )
        if report["joint_limit_excess_rad"]:
            print(f"[warning] unclipped joint limit excess: {report['joint_limit_excess_rad']}")
    (args.output_dir / SELECTION_NAME).write_text(
        yaml.safe_dump({"selected_files": selected_files}, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "format": "InstinctMJ retargeted (base quaternion wxyz)",
        "input_fps": args.input_fps,
        "output_fps": args.output_fps,
        "root_lateral_sign": args.root_lateral_sign,
        "root_yaw_sign": args.root_yaw_sign,
        "trim_start_frames": args.trim_start_frames,
        "keep_seconds": args.keep_seconds,
        "joint_names": list(CASBOT02_PARKOUR_JOINT_NAMES),
        "total_frames": sum(report["output_frames"] for _, report in converted),
        "missing_from_source_selection": missing,
        "motions": [report for _, report in converted],
    }
    (args.output_dir / "conversion_report.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if missing:
        print(f"[selection] Missing source motions, excluded from generated selection: {missing}")
    print(
        f"[ready] {len(selected_files)} motions, {manifest['total_frames']} frames; {args.output_dir / SELECTION_NAME}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "parkour_motion_reference")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--input-fps", type=int, default=1000)
    parser.add_argument("--output-fps", type=int, default=50)
    parser.add_argument("--trim-start-frames", type=int, default=0, help="Number of source frames to remove")
    parser.add_argument("--keep-seconds", type=float)
    parser.add_argument("--root-lateral-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--root-yaw-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--pitch-offset", choices=("auto", "none", "pi/2"), default="auto")
    parser.add_argument(
        "--allow-joint-limit-violations",
        action="store_true",
        help="Keep out-of-range source joint angles unchanged and report them",
    )
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("Use a separate output directory to preserve the source selection YAML")
    try:
        convert_directory(args)
    except ValueError as exc:
        parser.exit(1, f"Conversion failed: {exc}\n")


if __name__ == "__main__":
    main()
