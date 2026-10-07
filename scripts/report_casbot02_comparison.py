"""Summarize paired Play outcomes, including approach failure and flat controls."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

LABELS = ["success", "fall", "stuck", "off_course", "timeout"]


def counts(rows):
    c = Counter(r["outcome"] for r in rows)
    return [c[label] for label in LABELS]


def check_crossings(root, rows, protocol):
    """Count upright stair completion separately from stopping at the goal."""
    traces = np.load(root / "trajectories.npz")
    qpos, dt = traces["qpos"], float(traces["dt"])
    model = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
    data = mujoco.MjData(model)
    feet = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{name}") for name in ["leg_l6_link", "leg_r6_link"]
    ]
    result = []
    for i, row in enumerate(rows):
        meta = protocol["heightfields"][row["height_index"]]
        up = row["direction"] == "up"
        limit = meta["center_xy"][0] + (meta["plateau_left_x"] + 0.1 if up else meta["last_step_x"] + 0.15)
        consecutive, longest, first, passed_at = 0, 0.0, None, None
        for frame in range(len(qpos)):
            time_s = frame * dt + protocol["step_dt_s"]
            if time_s > row["time_s"] + 0.001:
                break  # Never count the held pose after an outcome.
            q = qpos[frame, i]
            ok = False
            if row["height_m"] > 0 and q[0] > limit - 0.3:
                data.qpos[:] = q
                mujoco.mj_kinematics(model, data)
                _, x, y, _ = q[3:7]
                floor = meta["top_m"] if up else meta["bottom_m"]
                ok = (
                    np.all(data.xpos[feet, 0] > limit)
                    and 1 - 2 * (x * x + y * y) > math.cos(0.6)
                    and q[2] - floor > 0.45
                )
            consecutive = consecutive + 1 if ok else 0
            if ok and first is None:
                first = time_s
            longest = max(longest, max(0, consecutive - 1) * dt)
            if longest >= 1 - 1e-6 and passed_at is None:
                passed_at = time_s
        result.append(
            dict(
                case_id=row["case_id"],
                checkpoint=row["checkpoint"],
                height_m=row["height_m"],
                direction=row["direction"],
                speed=row["speed"],
                crossed_upright_for_1s=longest >= 1 - 1e-6,
                max_consecutive_s=longest,
                first_crossing_s=first,
                passed_at_s=passed_at,
            )
        )
    (root / "crossings.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("logs/analysis/casbot02_checkpoint_comparison"))
    args = parser.parse_args()
    root = args.directory
    rows = json.loads((root / "results.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    diagnostics = json.loads((root / "diagnostics.json").read_text())
    crossings = check_crossings(root, rows, protocol)
    for r in rows:
        m = protocol["heightfields"][r["height_index"]]
        # Counts proximity to stair entry, not successful contact or ascent.
        entry_distance = 0.65 - r["x_offset"] if r["direction"] == "up" else -m["plateau_left_x"] - r["x_offset"]
        r["reached_stair_approach"] = r["height_m"] > 0 and r["max_progress_m"] >= entry_distance - 0.25
    summary = {}
    for cp in [2000, 8000]:
        stairs = [r for r in rows if r["checkpoint"] == cp and r["height_m"] > 0]
        flat = [r for r in rows if r["checkpoint"] == cp and r["height_m"] == 0]
        summary[str(cp)] = dict(
            stairs=dict(
                n=len(stairs),
                counts=dict(zip(LABELS, counts(stairs))),
                reached_approach=sum(r["reached_stair_approach"] for r in stairs),
                stuck_before_approach=sum(r["outcome"] == "stuck" and not r["reached_stair_approach"] for r in stairs),
                upright_stair_crossings=sum(c["crossed_upright_for_1s"] for c in crossings if c["checkpoint"] == cp),
            ),
            flat=dict(
                n=len(flat),
                counts=dict(zip(LABELS, counts(flat))),
                mean_max_progress_m=float(np.mean([r["max_progress_m"] for r in flat])),
            ),
        )
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    crossing_lines = []
    for cp in [2000, 8000]:
        details = []
        for direction in ["up", "down"]:
            selected = [
                c for c in crossings if c["checkpoint"] == cp and c["direction"] == direction and c["height_m"] > 0
            ]
            passed = sum(c["crossed_upright_for_1s"] for c in selected)
            details.append(
                f'{"上" if direction == "up" else "下"}台阶 {passed}/{len(selected)}（{passed/len(selected):.1%}）'
            )
        crossing_lines.append(str(cp) + "：" + "，".join(details) + "。")
    passing = [c for c in crossings if c["crossed_upright_for_1s"]]
    for group in sorted(set((c["checkpoint"], c["height_m"], c["direction"], c["speed"]) for c in passing)):
        cp, h, direction, speed = group
        passed = sum(
            c["checkpoint"] == cp and c["height_m"] == h and c["direction"] == direction and c["speed"] == speed
            for c in passing
        )
        selected = [
            c for c in crossings if c["checkpoint"] == cp and c["height_m"] == h and c["direction"] == direction
        ]
        crossing_lines.append(
            f"{cp} 的 {passed} 次跨越来自 {h*100:.0f}cm"
            f" {'上' if direction == 'up' else '下'}台阶、{speed}m/s；该高度与方向共 {len(selected)} 次测试。"
        )
    lines = [
        "# CASBOT02：model_2000 与 model_8000 确定性 Play 比较",
        "",
        (
            f"检查点：`{protocol.get('run_dir', 'logs/instinct_rl/casbot02_parkour/2026-10-05_20-54-33')}`。"
            "结果来自实际策略均值推理与原生 InstinctRlEnv 仿真；不采样动作噪声。"
        ),
        "",
        (
            f"每个检查点测试 {summary['2000']['stairs']['n']} 次台阶通行：5、10、15、20 cm，每档分别上/下台阶，"
            f"速度为 {protocol['speeds_m_s']} m/s，每种组合 {protocol['variants_per_height_direction_speed']} 个配对起点。"
            f"另测 {summary['2000']['flat']['n']} 次平地控制。每次最多 {protocol['duration_s']:g} 秒"
            "（含最初 1 秒站立与历史填充）。"
        ),
        "",
        (
            "这次采用规则、无墙、无 Perlin 扰动的训练地形生成器楼梯，每侧 5 级，踏面 35 cm。初始位置、偏航、目标、"
            "控制时间、摄像机和动作配置一致；观测扰动关闭，电机延迟固定 5 ms，深度历史延迟固定 1 个策略帧。"
            "保留相机裁剪、归一化、历史及原来的视觉编码器。"
        ),
        "",
        "配对初始观测最大差："
        + str(protocol["initial_obs_max_pair_difference"])
        + "；运行中直接核对策略输入历史的最新速度命令，最大误差："
        + str(max((d["command_observation_max_error"] for d in diagnostics), default=0))
        + "。",
        "",
        "## 判定规则",
        "",
        (
            "- **通过**：进入目标 25 cm 内，双脚踝越过最后一级边界，倾斜小于 0.6 rad，水平速度小于 0.25 m/s，持续 1"
            " 秒。上下台阶分别判定。"
        ),
        "- **跌倒**：机身倾斜超过 1 rad，或机身距当前位置真实地表低于 45 cm。",
        "- **卡住**：持续有前进命令（vx > 0.30 m/s），5 秒前进不足 15 cm，且尚未到目标。最早在 6.52 秒判定。",
        "- **偏离路线**：横向偏离通行中心超过 60 cm。",
        f"- **超时**：{protocol['duration_s']:g} 秒内未达到以上终止条件。",
        "- 判定互斥；同一时刻按跌倒、偏离、通过、卡住的次序记录。没有把存活时间作为通过条件。",
        "",
        "## 台阶跨越与完整目标通行分开统计",
        "",
        (
            "**台阶跨越**：双脚踝越过最后一级边界、机身倾斜小于 0.6 rad、距最终平台高度大于 45 cm，连续保持至少 1"
            " 秒；不要求停到指定目标。使用保存的 10 Hz 轨迹核查，不计结局后的定格帧。"
        ),
        "",
        "\n\n".join(crossing_lines),
        "",
        "跨越是额外的能力指标，不与下表终止结局相加。完整目标通行还要求在指定目标区域低速稳定 1 秒。",
        "",
        "## 完整目标通行的终止结局",
        "",
        "|检查点|次数|目标通过|跌倒|卡住|偏离|超时|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cp in [2000, 8000]:
        ss = summary[str(cp)]["stairs"]
        n = ss["n"]
        vals = [f'{ss["counts"][l]} ({ss["counts"][l]/n:.1%})' for l in LABELS]
        lines.append("|" + str(cp) + "|" + str(n) + "|" + "|".join(vals) + "|")
    lines += [
        "",
        "## 按台阶高度与方向",
        "",
        "|高度|方向|检查点|次数|通过|跌倒|卡住|偏离|超时|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in [0.05, 0.1, 0.15, 0.2]:
        for direction in ["up", "down"]:
            for cp in [2000, 8000]:
                selected = [
                    r for r in rows if r["checkpoint"] == cp and r["height_m"] == h and r["direction"] == direction
                ]
                lines.append(
                    "|"
                    + f"{h*100:.0f}cm|"
                    + ("上" if direction == "up" else "下")
                    + "|"
                    + str(cp)
                    + "|"
                    + str(len(selected))
                    + "|"
                    + "|".join(map(str, counts(selected)))
                    + "|"
                )
    lines += [
        "",
        "## 平地控制与接近台阶情况",
        "",
        "|检查点|速度|次数|通过|跌倒|卡住|偏离|超时|最大前进距离均值|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cp in [2000, 8000]:
        for speed in protocol["speeds_m_s"]:
            flat = [r for r in rows if r["checkpoint"] == cp and r["height_m"] == 0 and r["speed"] == speed]
            lines.append(
                "|"
                + f"{cp}|{speed}|{len(flat)}|"
                + "|".join(map(str, counts(flat)))
                + "|"
                + f'{np.mean([r["max_progress_m"] for r in flat]):.3f}m|'
            )
    for cp in [2000, 8000]:
        s = summary[str(cp)]["stairs"]
        lines += [
            "",
            (
                f'{cp}：{s["reached_approach"]}/{s["n"]} 次曾到台阶入口前 25 cm'
                f' 范围；{s["stuck_before_approach"]} 次卡住发生在进入该范围之前。接近入口仅说明机器人走到了楼梯前，'
                "不能算已上/下台阶。"
            ),
            "",
        ]
    lines += [
        "## 解释与限制",
        "",
        (
            "2000 的低跌倒率需要结合平地和位移看：若一直不走，它可以存活很久，但没有完成通行。8000"
            " 是否改进应以实际推进、通过率和跌倒共同判断，不能只比较 episode_length 或训练 reward。"
        ),
        "",
        (
            "本次推理不采样策略 std，因此高 Mean action noise std 不会直接成为这次 Play 的动作噪声。"
            "这些结果反映策略均值本身的表现；探索期的生存和 reward 不能保证均值策略已经学会跑酷。"
        ),
        "",
        "**地形配置更正**：此前设置的 `hfield_resolution=0.10` 未作用到原生高度场路径。本次实际编译高度场为"
        " 114×114，间距约 "
        + f'{protocol["heightfields"][1]["spacing_m"][0]*100:.2f}'
        + " cm，而非 10 cm。两个检查点使用相同的真实 Play 地形。本次没有修改正在训练的碰撞或奖励配置，理想 10 cm"
        " 碰撞网格的独立测试不能代替这次实际模型检查。",
        "",
        (
            "这是固定控制器、规则楼梯及名义动力学下的一次配对比较。"
            "起点变体不是独立训练种子；比例不应外推为真实世界通过率。卡住判定会提前结束低速推进案例，"
            "不能证明它在任意长时间下永远不能通过。每次尝试遇到首次结局即停止计分，不计后续恢复。"
        ),
        "",
        "## 文件与复现",
        "",
        (
            "- [完整每次试验记录](trials.csv)、[协议](protocol.json)、[输入诊断](diagnostics.json)、[汇总"
            " JSON](summary.json)。"
        ),
        (
            "- [平地 0.8m/s 视频](flat_08.mp4)、[上 10cm 台阶视频](up_10cm_08.mp4)、[下 5cm"
            " 跨越视频](down_05cm_08.mp4)、[下 10cm"
            " 台阶视频](down_10cm_08.mp4)。左侧 2000、右侧 8000，同一初始条件；首次结局后画面定格并标注。"
        ),
        (
            "- 平地和上台阶视频按 8000 最大前进距离选择；下 5cm 从稳定跨越案例中选最大前进距离；下 10cm 选择 8000"
            " 最早跌倒案例。选择记录见"
            " `video_selection.json`，视频只作现象演示，统计使用全部样本。"
        ),
        "- `exploratory_04_08/` 是第一轮 0.4/0.8 m/s 探索测试，不混入上述正式结果。",
        "",
        "```bash",
        "uv run python scripts/compare_casbot02_checkpoints.py --variants 16 --speeds 0.5 0.8 --duration 20",
        "uv run python scripts/report_casbot02_comparison.py",
        "uv run python scripts/render_casbot02_comparison.py",
        "```",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    colors = ["#2ca02c", "#d62728", "#ffbf00", "#9467bd", "#808080"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, cp in zip(axes, [2000, 8000]):
        heights = [0.05, 0.1, 0.15, 0.2]
        groups = [[r for r in rows if r["checkpoint"] == cp and r["height_m"] == h] for h in heights]
        bottom = np.zeros(4)
        for label, color in zip(LABELS, colors):
            val = np.array([sum(r["outcome"] == label for r in rs) / len(rs) * 100 for rs in groups])
            ax.bar(np.arange(4), val, bottom=bottom, color=color, label=label)
            bottom += val
        ax.set_xticks(np.arange(4), ["5cm", "10cm", "15cm", "20cm"])
        ax.set_title(f"model_{cp} (64 trials / height)")
        ax.set_ylabel("Trials (%)")
        ax.set_ylim(0, 100)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Goal-stopped trial outcomes: 0.5 / 0.8 m/s, paired initial states")
    fig.tight_layout()
    fig.savefig(root / "outcomes.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
