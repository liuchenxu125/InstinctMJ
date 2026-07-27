"""CASBOT_02 23-DOF humanoid robot asset definitions for InstinctMJ.

Follows the same pattern as ``unitree_g1.py``.
"""

from __future__ import annotations

import os

import mujoco
from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityCfg

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

# ---------------------------------------------------------------------------
# MJCF path
# ---------------------------------------------------------------------------

CASBOT02_MJCF_PATH: str = os.path.join(
    __file_dir__, "resources/casbot_23dof/xml/CASBOT_02_23dof.xml"
)
CASBOT02_MESHES_DIR: str = os.path.join(
    __file_dir__, "resources/casbot_23dof/meshes"
)

# ---------------------------------------------------------------------------
# Joint names (MJCF / actuator order, 23 DOF).
# This order is what mjlab's entity will produce in its joint-position /
# joint-velocity tensors.  Keep every downstream buffer in this order.
# ---------------------------------------------------------------------------

CASBOT02_23DOF_JOINT_NAMES: tuple[str, ...] = (
    # left leg (6)
    "leg_l1_joint",   # 0  – hip pitch  (axis 0 1 0)
    "leg_l2_joint",   # 1  – hip roll   (axis 1 0 0)
    "leg_l3_joint",   # 2  – hip yaw    (axis 0 0 1)
    "leg_l4_joint",   # 3  – knee pitch (axis 0 1 0)
    "leg_l5_joint",   # 4  – ankle pitch(axis 0 1 0)
    "leg_l6_joint",   # 5  – ankle roll (axis 1 0 0)
    # right leg (6)
    "leg_r1_joint",   # 6  – hip pitch  (axis 0 1 0)
    "leg_r2_joint",   # 7  – hip roll   (axis 1 0 0)
    "leg_r3_joint",   # 8  – hip yaw    (axis 0 0 1)
    "leg_r4_joint",   # 9  – knee pitch (axis 0 1 0)
    "leg_r5_joint",   # 10 – ankle pitch(axis 0 1 0)
    "leg_r6_joint",   # 11 – ankle roll (axis 1 0 0)
    # waist (1)
    "waist_yaw_joint",# 12 – waist yaw  (axis 0 0 1)
    # left arm (5)
    "upper_left_1_joint",  # 13 – shoulder pitch (axis 0 1 0)
    "upper_left_2_joint",  # 14 – shoulder roll  (axis 1 0 0)
    "upper_left_3_joint",  # 15 – shoulder yaw   (axis 0 0 1)
    "upper_left_4_joint",  # 16 – elbow pitch    (axis 0 1 0)
    "upper_left_5_joint",  # 17 – wrist yaw      (axis 0 0 1)
    # right arm (5)
    "upper_right_1_joint", # 18 – shoulder pitch (axis 0 1 0)
    "upper_right_2_joint", # 19 – shoulder roll  (axis 1 0 0)
    "upper_right_3_joint", # 20 – shoulder yaw   (axis 0 0 1)
    "upper_right_4_joint", # 21 – elbow pitch    (axis 0 1 0)
    "upper_right_5_joint", # 22 – wrist yaw      (axis 0 0 1)
)

# ---------------------------------------------------------------------------
# Leg-only joint names (lower-body policy, 13 DOF)
# ---------------------------------------------------------------------------

CASBOT02_LEG_JOINT_NAMES: tuple[str, ...] = (
    "leg_l1_joint", "leg_l2_joint", "leg_l3_joint",
    "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
    "leg_r1_joint", "leg_r2_joint", "leg_r3_joint",
    "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
    "waist_yaw_joint",
)

# ---------------------------------------------------------------------------
# Body names used for AMP style reward / motion reference
# ---------------------------------------------------------------------------

CASBOT02_LEG_AMP_BODY_NAMES: tuple[str, ...] = (
    "torso",
    "leg_l2_link",   # left hip roll link (thigh)
    "leg_l4_link",   # left knee link (shank)
    "leg_l6_link",   # left ankle roll link (foot)
    "leg_r2_link",   # right hip roll link (thigh)
    "leg_r4_link",   # right knee link (shank)
    "leg_r6_link",   # right ankle roll link (foot)
    "waist_yaw_link",
)

# ---------------------------------------------------------------------------
# Symmetric augmentation – joint mapping (left ↔ right)
# ---------------------------------------------------------------------------

CASBOT02_SYMMETRIC_JOINT_MAPPING: tuple[int, ...] = (
    6,   # 0  leg_l1 → leg_r1
    7,   # 1  leg_l2 → leg_r2
    8,   # 2  leg_l3 → leg_r3
    9,   # 3  leg_l4 → leg_r4
    10,  # 4  leg_l5 → leg_r5
    11,  # 5  leg_l6 → leg_r6
    0,   # 6  leg_r1 → leg_l1
    1,   # 7  leg_r2 → leg_l2
    2,   # 8  leg_r3 → leg_l3
    3,   # 9  leg_r4 → leg_l4
    4,   # 10 leg_r5 → leg_l5
    5,   # 11 leg_r6 → leg_l6
    12,  # 12 waist_yaw → self
    18,  # 13 upper_left_1 → upper_right_1
    19,  # 14 upper_left_2 → upper_right_2
    20,  # 15 upper_left_3 → upper_right_3
    21,  # 16 upper_left_4 → upper_right_4
    22,  # 17 upper_left_5 → upper_right_5
    13,  # 18 upper_right_1 → upper_left_1
    14,  # 19 upper_right_2 → upper_left_2
    15,  # 20 upper_right_3 → upper_left_3
    16,  # 21 upper_right_4 → upper_left_4
    17,  # 22 upper_right_5 → upper_left_5
)

# ---------------------------------------------------------------------------
# Symmetric augmentation – sign reverse buffer
#   pitch-axis joints: sign = +1
#   roll / yaw-axis joints: sign = -1
# ---------------------------------------------------------------------------

CASBOT02_SYMMETRIC_JOINT_REVERSE_BUF: tuple[int, ...] = (
    1,   # 0  leg_l1 (hip pitch)
    -1,  # 1  leg_l2 (hip roll)
    -1,  # 2  leg_l3 (hip yaw)
    1,   # 3  leg_l4 (knee pitch)
    1,   # 4  leg_l5 (ankle pitch)
    -1,  # 5  leg_l6 (ankle roll)
    1,   # 6  leg_r1 (hip pitch)
    -1,  # 7  leg_r2 (hip roll)
    -1,  # 8  leg_r3 (hip yaw)
    1,   # 9  leg_r4 (knee pitch)
    1,   # 10 leg_r5 (ankle pitch)
    -1,  # 11 leg_r6 (ankle roll)
    -1,  # 12 waist_yaw (yaw)
    1,   # 13 upper_left_1 (shoulder pitch)
    -1,  # 14 upper_left_2 (shoulder roll)
    -1,  # 15 upper_left_3 (shoulder yaw)
    1,   # 16 upper_left_4 (elbow pitch)
    -1,  # 17 upper_left_5 (wrist yaw)
    1,   # 18 upper_right_1 (shoulder pitch)
    -1,  # 19 upper_right_2 (shoulder roll)
    -1,  # 20 upper_right_3 (shoulder yaw)
    1,   # 21 upper_right_4 (elbow pitch)
    -1,  # 22 upper_right_5 (wrist yaw)
)

# ---------------------------------------------------------------------------
# Symmetric augmentation – link mapping (for AMP body references)
# ---------------------------------------------------------------------------

CASBOT02_LEG_AMP_SYMMETRIC_LINK_MAPPING: tuple[int, ...] = (
    0,  # 0 torso → self
    4,  # 1 leg_l2 → leg_r2
    5,  # 2 leg_l4 → leg_r4
    6,  # 3 leg_l6 → leg_r6
    1,  # 4 leg_r2 → leg_l2
    2,  # 5 leg_r4 → leg_l4
    3,  # 6 leg_r6 → leg_l6
    7,  # 7 waist_yaw → self
)

# ---------------------------------------------------------------------------
# spec function
# ---------------------------------------------------------------------------


def get_casbot02_spec() -> mujoco.MjSpec:
    """Load the CASBOT_02 23-DOF XML as MjSpec."""
    return mujoco.MjSpec.from_file(CASBOT02_MJCF_PATH)


# ---------------------------------------------------------------------------
# Initial state (standing pose matching the home keyframe)
# ---------------------------------------------------------------------------

CASBOT02_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.92),
    joint_pos={
        "leg_l1_joint": -0.185,
        "leg_l2_joint": 0.0,
        "leg_l3_joint": 0.0,
        "leg_l4_joint": 0.36,
        "leg_l5_joint": -0.175,
        "leg_l6_joint": 0.0,
        "leg_r1_joint": -0.185,
        "leg_r2_joint": 0.0,
        "leg_r3_joint": 0.0,
        "leg_r4_joint": 0.36,
        "leg_r5_joint": -0.175,
        "leg_r6_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "upper_left_1_joint": 0.0,
        "upper_left_2_joint": 0.0,
        "upper_left_3_joint": 0.0,
        "upper_left_4_joint": 0.0,
        "upper_left_5_joint": 0.0,
        "upper_right_1_joint": 0.0,
        "upper_right_2_joint": 0.0,
        "upper_right_3_joint": 0.0,
        "upper_right_4_joint": 0.0,
        "upper_right_5_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)

# ---------------------------------------------------------------------------
# Actuator parameters (from casbot02_constants.py, adapted to BuiltinPdActuatorCfg)
# ---------------------------------------------------------------------------

# Natural frequency and damping ratio for PD gains
_NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
    return armature * _NATURAL_FREQ ** 2


def _damping(armature: float) -> float:
    return 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ


# Per-joint-group actuator configs with motor communication delay
# Following the same pattern as beyondmimic_g1_29dof_delayed_actuator_cfgs.

_DELAY_RESET_ONLY_PERIOD = 1_000_000

CASBOT02_DELAYED_LEG_HEAVY = BuiltinPdActuatorCfg(
    target_names_expr=(
        "leg_l1_joint", "leg_l2_joint", "leg_l4_joint",
        "leg_r1_joint", "leg_r2_joint", "leg_r4_joint",
    ),
    stiffness=_stiffness(0.07),
    damping=_damping(0.07),
    effort_limit=120.0,
    armature=0.07,
    delay_min_lag=0,
    delay_max_lag=2,
    delay_update_period=_DELAY_RESET_ONLY_PERIOD,
    delay_per_env_phase=False,
)

CASBOT02_DELAYED_LEG_LIGHT = BuiltinPdActuatorCfg(
    target_names_expr=(
        "leg_l3_joint", "leg_l5_joint", "leg_l6_joint",
        "leg_r3_joint", "leg_r5_joint", "leg_r6_joint",
    ),
    stiffness=_stiffness(0.029),
    damping=_damping(0.029),
    effort_limit=80.0,
    armature=0.029,
    delay_min_lag=0,
    delay_max_lag=2,
    delay_update_period=_DELAY_RESET_ONLY_PERIOD + 1,
    delay_per_env_phase=False,
)

CASBOT02_DELAYED_WAIST_YAW = BuiltinPdActuatorCfg(
    target_names_expr=("waist_yaw_joint",),
    stiffness=_stiffness(0.0245),
    damping=_damping(0.0245),
    effort_limit=60.0,
    armature=0.0245,
    delay_min_lag=0,
    delay_max_lag=2,
    delay_update_period=_DELAY_RESET_ONLY_PERIOD + 2,
    delay_per_env_phase=False,
)

CASBOT02_DELAYED_ARM_HEAVY = BuiltinPdActuatorCfg(
    target_names_expr=(
        "upper_left_1_joint", "upper_left_2_joint", "upper_left_4_joint",
        "upper_right_1_joint", "upper_right_2_joint", "upper_right_4_joint",
    ),
    stiffness=_stiffness(0.033),
    damping=_damping(0.033),
    effort_limit=40.0,
    armature=0.033,
    delay_min_lag=0,
    delay_max_lag=2,
    delay_update_period=_DELAY_RESET_ONLY_PERIOD + 3,
    delay_per_env_phase=False,
)

CASBOT02_DELAYED_ARM_LIGHT = BuiltinPdActuatorCfg(
    target_names_expr=(
        "upper_left_3_joint", "upper_left_5_joint",
        "upper_right_3_joint", "upper_right_5_joint",
    ),
    stiffness=_stiffness(0.0245),
    damping=_damping(0.0245),
    effort_limit=30.0,
    armature=0.0245,
    delay_min_lag=0,
    delay_max_lag=2,
    delay_update_period=_DELAY_RESET_ONLY_PERIOD + 4,
    delay_per_env_phase=False,
)

casbot02_23dof_delayed_actuator_cfgs: tuple = (
    CASBOT02_DELAYED_LEG_HEAVY,
    CASBOT02_DELAYED_LEG_LIGHT,
    CASBOT02_DELAYED_WAIST_YAW,
    CASBOT02_DELAYED_ARM_HEAVY,
    CASBOT02_DELAYED_ARM_LIGHT,
)

# ---------------------------------------------------------------------------
# Action scale (0.25 * effort_limit / stiffness, same formula as G1)
# ---------------------------------------------------------------------------

casbot02_action_scale: dict[str, float] = {}
for _act in casbot02_23dof_delayed_actuator_cfgs:
    _effort = _act.effort_limit
    _stiff = _act.stiffness
    if _effort is None or _stiff == 0.0:
        continue
    for _name in _act.target_names_expr:
        casbot02_action_scale[_name] = 0.25 * _effort / _stiff

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CASBOT02_23DOF_JOINT_NAMES",
    "CASBOT02_DELAYED_ARM_HEAVY",
    "CASBOT02_DELAYED_ARM_LIGHT",
    "CASBOT02_DELAYED_LEG_HEAVY",
    "CASBOT02_DELAYED_LEG_LIGHT",
    "CASBOT02_DELAYED_WAIST_YAW",
    "CASBOT02_INIT_STATE",
    "CASBOT02_LEG_AMP_BODY_NAMES",
    "CASBOT02_LEG_AMP_SYMMETRIC_LINK_MAPPING",
    "CASBOT02_LEG_JOINT_NAMES",
    "CASBOT02_MESHES_DIR",
    "CASBOT02_MJCF_PATH",
    "CASBOT02_SYMMETRIC_JOINT_MAPPING",
    "CASBOT02_SYMMETRIC_JOINT_REVERSE_BUF",
    "casbot02_23dof_delayed_actuator_cfgs",
    "casbot02_action_scale",
    "get_casbot02_spec",
]
