# Parkour Task

## CASBOT02 Joint Configuration

`Instinct-Parkour-Target-Amp-CASBOT02-v0` and its Play task use the
`CASBOT_02_18dof.xml` model with 18 controlled joints: 12 leg joints
and three joints per arm (shoulder pitch/roll and elbow pitch).
The waist yaw joint (`waist_yaw_joint`) is rigidly fixed at zero.
Both shoulder yaw joints (`upper_left_3_joint`, `upper_right_3_joint`) and
both wrist yaw joints (`upper_left_5_joint`, `upper_right_5_joint`) are rigidly
fixed at zero. Wrist pitch/roll were already fixed in the 23-DOF source model.
Training, motion-reference kinematics, playback, and
`scripts/sim2sim_casbot02_parkour.py` use the same 18-joint order.
Existing 19- or 23-action checkpoints require retraining and re-exporting.
Reference motion files may contain the original 23 CASBOT02 joint names;
the motion loader selects the 18 active joints by name.

The hip default-pose penalty (`joint_deviation_hip`, weight `-0.5`) applies
only to hip roll/yaw (`leg_l2/l3_joint`, `leg_r2/r3_joint`). Hip pitch is free
of this penalty so the policy can flex the hips when stepping onto obstacles.

CASBOT02 camera pose matches InstinctLab: offset `(0.080, 0.0, 0.1495)` meters
and 60 degrees downward pitch, relative to `torso` (InstinctLab's `base_link`).
The raw field of view matches InstinctLab's `94 x 68` degrees at `64 x 36` pixels.
Cropping `(top, bottom, left, right) = (18, 0, 16, 16)` removes the upper half
and 16 columns on each side, matching the RPO/G1 parkour crop. The retained
region is `depth[18:36, 16:48]`, producing `18 x 32` pixels without resizing.
The quaternion represents a 60-degree rotation around Y: the optical axis
points 60 degrees below forward horizontal, or 30 degrees from downward -Z.
Training, Play, and sim2sim share these camera settings from `assets/casbot_02.py`.
Checkpoints trained with the former center crop `(9, 9, 16, 16)` receive a
different image region under this setting; their earlier evaluation used the
former crop and should not be treated as an evaluation of the new setting.

## CASBOT02 Shell Contacts

The 18-DOF parkour XML retains this project's joint definitions, body transforms,
explicit masses and inertias, and visual meshes. Its physical collisions follow
InstinctLab's `CASBOT02_7dof/urdf/CASBOT02_ENCOS_18dof_shell.urdf`.
InstinctLab's `CASBOT02_18DOF_CFG` sets `replace_cylinders_with_capsules=True`,
so each URDF cylinder becomes a MuJoCo capsule with the same local origin,
rotation, radius, and cylindrical segment length. URDF leg-link names map to
`leg_l1` through `leg_l6` / `leg_r1` through `leg_r6`; `base_link` maps to `torso`.

| Part | Physical collision shapes |
|---|---|
| Base (`torso`) | One sphere, radius 0.09 m, center `(0, 0, 0.065)` |
| Leg pelvic-roll and knee links | Four capsules total, radius 0.06 m, segment length 0.24 m, center `(0.025, 0, -0.12)` |
| Feet | Seven capsules per foot, radius 0.01 m, segment lengths `0.20, 0.24, 0.26, 0.26, 0.26, 0.24, 0.20` m |
| Waist/upper torso | Four capsules, matching the URDF origins and dimensions |
| Shoulder-yaw and elbow links | Four capsules total, radius 0.045 m, segment length 0.17 m |

There are 27 physical geoms: 26 capsules and one sphere. All other geoms are
visual only; physical mesh collisions are removed. Waist, shoulder yaw and
wrists remain fixed, with 12 leg and 6 arm joints active. No new contact exclusions
are introduced. Training, Play, motion-reference FK and sim2sim load the same
local XML; MuJoCo and PhysX still have different contact solvers.

The seven foot axes are parallel to X, at X=0.035 m, Z=-0.054 m, and
Y=`-0.040, -0.026, -0.013, 0, 0.013, 0.026, 0.040` m. The configured sole height
is 0.064 m (`feet_at_plane.height_offset`), matching InstinctLab.
The reward volume grid also matches InstinctLab: X `[-0.025, 0.12]`,
Y `[-0.03, 0.03]`, Z `[-0.069, -0.029]` m with `10 x 5 x 2` samples per foot.
The original foot STL remains visible, with capsule geometry determining support.

Physical geoms use the existing collision material class (`condim=3`). Feet retain
`priority=1` so their randomized friction governs terrain contacts, including
values below the terrain's coefficient. Force sensing aggregates the seven geoms
per foot; the existing 1 N contact classification, force history, air-time tracking,
foot sliding and foot orientation rewards remain active. Knee contacts now count
in undesired-contact rewards because the knees have physical capsules.

**Heightfield sampling correction:** `hfield_resolution=0.10` is still present in
CASBOT02's generator configuration, but the native heightfield path currently uses
`horizontal_scale=0.07`. Compiled 8 m tiles have 114 samples per axis, giving an
actual grid interval of about 0.07079646 m. No terrain resolution change was made
as part of this collision alignment. The earlier complete-mesh stress results at
ideal 10 cm spacing are historical, not evidence about the current shell model.

The shell model was tested against flat, rough and stair heightfields at its
actual 7.08 cm spacing with 192 deterministic and 128 random poses per geom.
This includes rotations, grid offsets and penetrations up to 15 cm.
The four large leg capsules produced 200 per-pair overflow warnings over the
25,920 geom/terrain/pose checks; the feet, sphere, waist and arm capsules produced
none. Primitive collisions therefore do not guarantee that every deeply penetrating
pose avoids MuJoCo-Warp's per-pair heightfield limit. Reproduce with:

```bash
uv run python scripts/check_casbot02_hfield_contacts.py \
  --resolution 0.07079646017699115 --random-poses 128 \
  --report /tmp/casbot02_shell_heightfield_contacts.json
```

The per-geom report is saved in
`logs/analysis/casbot02_isaac_shell_collision/heightfield_contacts.json`.
Existing checkpoints were trained with different collision geometry; their prior
Play comparison does not evaluate this new model. Restart an existing training
process to load the new asset and reward sampling settings.

## CASBOT02 Training Randomization

The mass, center-of-mass, and PD ranges match the effective configuration in
`~/Desktop/HANDOFF/src/wbc_mjlab/casbot02_config.py::_apply_casbot02_dr`, including
its inherited velocity-task COM term. Camera installation errors follow the RPO
parkour camera term. PD randomization additionally includes all six active arm
joints, as requested. The configuration is in
`config/casbot02/casbot02_parkour_target_amp_cfg.py`.

| Parameter | Range / behavior | Sampling |
| --- | --- | --- |
| Torso mass | Nominal mass plus uniform `[-4, 4]` kg | Startup, per environment |
| All non-torso link masses | Nominal mass times uniform `[0.8, 1.2]`, independently per link | Startup, per environment |
| Torso COM | Local X/Y offsets `[-0.025, 0.025]` m, Z `[-0.03, 0.03]` m | Startup, per environment |
| Leg and arm PD gains | Kp and Kd independently scaled by uniform `[0.85, 1.15]`, on all 18 active joints (12 leg + 6 arm) | Startup, per environment and joint |
| Camera translation | Local X/Y/Z errors `[-0.03, 0.03]` m relative to the nominal attachment | Startup, per environment |
| Camera rotation | Local Euler roll/pitch/yaw errors `[-3, 3]` degrees around the nominal Euler angles | Startup, per environment |
| Robot geom sliding friction | Uniform `[0.3, 1.6]`; one coefficient shared by the robot's geoms within each environment | Startup, per environment |
| Motor command delay | Integer `0`, `1`, or `2` physics steps: `0`, `5`, or `10` ms; leg-heavy, leg-light, and arm-heavy groups | Reset, held for the episode |
| Depth-image delay | Configured uniform floating delay `[0, 1]` frame; see the indexing limitation below | Reset |
| Policy base angular velocity noise | Uniform additive `[-0.2, 0.2]` rad/s before scaling by `0.25` | Every observation |
| Policy projected-gravity noise | Uniform additive `[-0.05, 0.05]` per component | Every observation |
| Policy joint position noise | Uniform additive `[-0.01, 0.01]` rad | Every observation |
| Policy joint velocity noise | Uniform additive `[-0.5, 0.5]` rad/s before scaling by `0.05` | Every observation |
| Root initial pose | X/Y offsets `[-0.1, 0.1]` m and yaw `[-0.1, 0.1]` rad | Reset |
| Root initial velocity | Each linear component `[-0.2, 0.2]` m/s; each angular component `[-0.2, 0.2]` rad/s | Reset |
| Joint initial pose | Nominal joint positions plus `[-0.15, 0.15]` rad; initial joint velocities are zero | Reset |

Startup samples remain fixed across episode resets. Mass updates intentionally
leave the inertia tensors unchanged, matching HANDOFF's `body_mass` implementation.
Friction uses mjlab's decorated `dr.geom_friction` term to expand the model field
per environment. Its range describes robot geom coefficients; the final contact
coefficient also depends on MuJoCo's material combination with the terrain.
The foot geoms have higher priority than terrain, so their sampled coefficients
are used directly for foot-terrain contacts.

The existing depth-delay implementation samples a continuous value and then
converts the resulting history index to an integer. With this configuration,
almost every positive sample selects the previous frame (about 20 ms), rather
than producing an evenly sampled choice between zero and one frame. This behavior
is retained here. The camera keeps only its existing fixed center crop,
`3 x 3` Gaussian blur (`sigma=1`), and normalization to a `0–2.5` m depth range.
Depth-scale errors, stereo holes, random convolution, fractal depth noise, and
random pixel noise are not enabled. The fixed blur is not random Gaussian noise.

Task and AMP sampling also vary during training:

- Terrain: the seeded (`seed=0`) `10 x 20` generated grid includes rough ground,
  gaps, stairs up/down, boxes, and slopes; Perlin geometry, obstacle locations,
  flat target patches, and boundary walls are sampled at generation. Each side
  has wall probability `0.3`. Initial terrain levels are sampled from `0–5`,
  then adjusted by the terrain curriculum. Normal stair heights span `5–23` cm
  and high stairs span `5–45` cm across the full curriculum.
- Commands: target patches and velocity limits resample every `8–12` s; rough
  ground forward-speed limits span `0.45–1.0` m/s, obstacle/stair limits
  `0.45–0.8` m/s, lateral limits are zero, and yaw limits span `-1–1` rad/s.
  Actual commands track the sampled target; standing selection has probability
  `0.05`, and the rough-standing terrain commands zero velocity.
- AMP reference: motion clips and start times are sampled; starts range from
  `0–90%` of clip duration, and left/right mirroring is sampled with probability
  `0.5`. Reference frame spacing remains fixed at `0.02` s.

No external pushes, armature randomization, joint-friction randomization,
joint-default-position calibration errors, encoder bias, or restitution
randomization are enabled. Play omits the added mass/COM/PD/camera terms and
material randomization, but retains the pre-existing observation/delay and
root-reset behavior.

Run `uv run python -m unittest discover -s scripts/tests -v` to check the
CASBOT02 randomization contract, data conversion, and the existing sim2sim tests.

## CASBOT02 Motion Data

Put CASBOT02 `.data` CSV recordings in `Datasets/casbot02`, then run:

```bash
uv run python scripts/casbot02_data_to_npz.py
uv run instinct-train Instinct-Parkour-Target-Amp-CASBOT02-v0
```

The converter follows `amp_mjlab/scripts/casbot02_data_to_npz.py`'s source
conventions and downsamples from 1000 Hz to 50 Hz. It preserves the first numeric
frame, detects optional text headers, and rejects damaged frames. Positions are
in meters and angles in radians. Root `(x, y, z)` is `(column 1, column 0, column 2)`;
raw pitch/roll/yaw is in columns 3/4/5, with yaw negated. Upright pitch is
automatically recognized as either zero or pi/2 independently for each clip.
Columns here are zero-based. CLI flags permit explicit pitch/axis overrides,
start trimming, shorter clips, and alternative input/output sample rates with
an integer downsample ratio. Root turns, lateral motion, and height changes are
preserved, including stair ascent/descent.

The converter maps leg columns 12–23 and arm columns 24/25/27/31/32/34 to the
current 18-joint model, skipping all fixed joints. It checks joint order and
limits against `CASBOT_02_18dof.xml`; out-of-range angles fail by default and
are never silently clipped. Output filenames end in `_retargeted.npz`, with
`framerate`, `joint_names`, `joint_pos`, `base_pos_w`, and `base_quat_w` (wxyz).
The InstinctMJ motion loader computes joint/base velocities and link kinematics
from these arrays, using its configured velocity-estimation method.

Outputs go to `Datasets/casbot02/parkour_motion_reference`, matching the task's
default dataset path. The generated `parkour_motion_without_run.yaml` lists
only converted recordings; `conversion_report.json` records frame counts,
durations, pitch offsets, joint-limit checks, and missing source selections.
Original `.data` files and their source YAML are preserved. Rerunning conversion
replaces the generated files and selection in the output directory.

The supplied dataset contains five clips: upstairs, downstairs, left/right
turning, and forward walking, totaling 2624 frames at 50 Hz. Its source selection
also names `后退_retargeted.npz`, but no corresponding `.data` recording is
present, so that clip is excluded from the generated selection. Add its source
recording and rerun conversion to include it. These are reference motions for
AMP; obstacle and terrain curricula remain configured separately in the task.

## Basic Usage Guidelines

### Parkour Task

**Task IDs:**
- `Instinct-Parkour-Target-Amp-G1-v0` (train)
- `Instinct-Parkour-Target-Amp-G1-Play-v0` (play)

1. Go to `config/g1/g1_parkour_target_amp_cfg.py` and set the `path` and `filtered_motion_selection_filepath` in `AmassMotionCfg` to the reference motion you want to use.

   ```python
   path: str = os.path.expanduser("~/your/path/to/parkour_motion_reference")
   filtered_motion_selection_filepath: str | None = os.path.join(
       path,
       "parkour_motion_without_run.yaml",
   )
   ```

   Keep the selected motion `.npz` files and the selection `.yaml` aligned with the same dataset root unless you intentionally split them.

2. Train the policy:
```bash
instinct-train Instinct-Parkour-Target-Amp-G1-v0
```

3. Play trained policy (`--load-run` must be provided, absolute path is recommended, or use `--agent random` to visualize an untrained policy):

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name>
```

4. Export trained policy (`--load-run` must be provided, absolute path is recommended):

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --export-onnx True
```

5. Use the exported ONNX policy for play:

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --use-onnx True
```

## G1 MuJoCo Sim2Sim

`scripts/sim2sim_g1_parkour.py` runs the exported depth encoder and actor in
single-robot CPU MuJoCo. Its workflow is based on
[roboparty_train's parkour sim2sim](https://github.com/liuchenxu125/roboparty_train/blob/main/robolab/scripts/mujoco/sim2sim_rpo_parkour.py).
Robot joint order, shoes, default pose, PD controls, action scales, observation
scales, and camera settings come from this repository's G1 parkour configuration.
The policy uses 29 actions, 768 proprioceptive values (per-component histories),
and an `8 x 18 x 32` depth stack. Camera delay can be fixed to zero or one frame.
Training-time observation noise, motor lag randomization, pushes, and material
randomization are omitted for deterministic deployment checks. No reference
motion dataset, ROS, or Isaac Lab runtime is needed.

First export a checkpoint (replace `<run_name>` with your saved training run):

```bash
uv run instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 \
  --load-run <run_name> --export-onnx True --num-envs 1 --viewer none --max-steps 1
```

Then open the up/platform/down stair scene:

```bash
uv run python scripts/sim2sim_g1_parkour.py \
  --model-dir logs/instinct_rl/g1_parkour/<run_name>/exported \
  --terrain stairs --depth-preview --duration 0
```

`--model-dir` accepts either a run directory or its `exported` subdirectory.
When omitted, the script finds the latest complete export under `g1_parkour`.
Both `actor.onnx` and `0-depth_encoder.onnx` must be present. Models exported
with fixed batch sizes are supported. If available, `metadata.json` is checked
for task, action count, and observation layout. Export using this repository's
current G1 configuration; legacy Isaac Lab policies have a different joint order.

In the MuJoCo window, W/S adjust forward speed, A/D adjust lateral speed, and
Q/E adjust turning speed in increments of 0.1. Space stops, R resets the robot
and observation histories, F toggles camera following, and C toggles chase view.
Commands ramp smoothly. The depth preview shows raw depth with the crop rectangle
beside the actual policy image. Close the MuJoCo window or press Ctrl+C to exit.

Other scenes: `--terrain flat`, `up`, `down`, or `obstacles`. The down scene spawns
the robot on the upper platform. Stair geometry is a deterministic test course,
not the full randomized training terrain; heights and widths can be changed with
`--step-height`, `--step-width`, `--num-stairs`, and `--stairs-width`.

For a headless check and optional tracking log:

```bash
uv run python scripts/sim2sim_g1_parkour.py \
  --model-dir logs/instinct_rl/g1_parkour/<run_name>/exported \
  --terrain flat --headless --duration 10 --command 0.5 0 0 \
  --log-file /tmp/g1_sim2sim_trace.npz
```

`--no-realtime` removes wall-clock waiting. `--zero-policy` exercises the model
and depth/observation pipeline without trained weights; it does not evaluate a
learned policy. CPU integration checks are available with:

```bash
uv run python -m unittest discover -s scripts/tests -v
```

If a policy walks on flat ground but stalls at the first stair, evaluate lower
stairs first (`--step-height 0.05 --command 0.5 0 0`) and increase height only
after it completes the course. Compare the same checkpoint in the training
environment on fixed-height stairs as well. Training includes stair terrains,
but mean reward and mean curriculum level across all terrain types do not
measure stair completion. AMP reference motions constrain movement style;
loading them does not guarantee the learned policy can climb a given height.
The runner applies all training MuJoCo options, including the `implicitfast`
integrator, rather than relying on the source XML's Euler default.

## Onboard Joint Order

If you deploy the exported ONNX policy in `instinct_onboard`, make sure `instinct_onboard/instinct_onboard/robot_cfgs.py` uses the `InstinctMJ` MuJoCo joint order instead of the older InstinctLab-style shoulder-first order.

Current `instinct_onboard` `sim_joint_names` order is shoulder-first:

```python
[
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]
```

`InstinctMJ` G1 MJCF order is:

```python
[
    "waist_pitch_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
```

When switching `instinct_onboard` to the `InstinctMJ` order, update the following in `instinct_onboard/instinct_onboard/robot_cfgs.py` together:

1. Replace `sim_joint_names` with the `InstinctMJ` order above.
2. Recompute `joint_map` from `real_joint_names` using the same order:

```python
joint_map = [real_joint_names.index(name) for name in sim_joint_names]
```

3. Reorder every array that is documented as "in simulation order" to match the new `sim_joint_names`, especially `joint_signs`, `joint_limits_high`, `joint_limits_low`, and `torque_limits`.

You can safely generate those arrays by name instead of editing indices by hand:

```python
old_sim_joint_names = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

def reorder_from_old(values, sim_joint_names):
    by_name = {name: values[i] for i, name in enumerate(old_sim_joint_names)}
    return [by_name[name] for name in sim_joint_names]
```

This matters because `instinct_onboard/instinct_onboard/agents/base.py` and `instinct_onboard/instinct_onboard/agents/parkour_agent.py` both parse observation and action tensors by iterating `sim_joint_names`. If the onboard order stays on the old InstinctLab convention while the exported `InstinctMJ` config/ONNX uses MJ order, joint observations, action offsets, action scales, and zero-action masks will all be misaligned.

## Common Options

- `--num-envs`: Number of parallel environments (default varies by task)
- `--load-run`: Run name to load checkpoint from for playing
- `--video`: Record training/playback videos
- `--export-onnx`: Export the trained model to ONNX format for onboard deployment during playing
- `--use-onnx`: Use the ONNX model for inference during playing
