"""CASBOT_02 parkour AMP task config.

Follows the G1 parkour pattern (``g1_parkour_target_amp_cfg.py``),
adapted for CASBOT_02 with 18 active joints and rigid waist/shoulder/wrist yaw.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    PinholeCameraPatternCfg,
    RayCastSensorCfg,
)
from mjlab.utils.noise import UniformNoiseCfg

import instinct_mj.envs.mdp as envs_mdp
import instinct_mj.tasks.parkour.mdp as parkour_mdp
from instinct_mj.assets.casbot_02 import (
    CASBOT02_DEPTH_CAMERA_CROP_REGION,
    CASBOT02_DEPTH_CAMERA_FOV_X_DEG,
    CASBOT02_DEPTH_CAMERA_FOV_Y_DEG,
    CASBOT02_DEPTH_CAMERA_LINK,
    CASBOT02_DEPTH_CAMERA_OFFSET_POS,
    CASBOT02_DEPTH_CAMERA_OFFSET_ROT,
    CASBOT02_FOOT_SOLE_HEIGHT,
    CASBOT02_LEG_AMP_BODY_NAMES,
    CASBOT02_LEG_AMP_SYMMETRIC_LINK_MAPPING,
    CASBOT02_PARKOUR_INIT_STATE,
    CASBOT02_PARKOUR_JOINT_NAMES,
    CASBOT02_PARKOUR_MJCF_PATH,
    CASBOT02_PARKOUR_SYMMETRIC_JOINT_MAPPING,
    CASBOT02_PARKOUR_SYMMETRIC_JOINT_REVERSE_BUF,
    casbot02_parkour_action_scale,
    casbot02_parkour_delayed_actuator_cfgs,
    get_casbot02_parkour_spec,
)
from instinct_mj.envs.manager_based_rl_env_cfg import InstinctLabRLEnvCfg
from instinct_mj.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinct_mj.motion_reference.motion_reference_cfg import MotionReferenceManagerCfg
from instinct_mj.motion_reference.utils import motion_interpolate_bilinear
from instinct_mj.sensors.contact_sensor import ForceThresholdContactSensorCfg
from instinct_mj.sensors.noisy_camera import NoisyGroupedRayCasterCameraCfg
from instinct_mj.sensors.volume_points import Grid3dPointsGeneratorCfg, VolumePointsCfg
from instinct_mj.tasks.parkour.config.parkour_env_cfg import (
    ROUGH_TERRAINS_CFG,
    ROUGH_TERRAINS_CFG_PLAY,
    _edit_parkour_scene_spec,
)
from instinct_mj.tasks.parkour.mdp.commands import PoseVelocityCommandCfg
from instinct_mj.terrains.terrain_importer_cfg import TerrainImporterCfg as InstinctTerrainImporterCfg
from instinct_mj.terrains.virtual_obstacle.edge_cylinder_cfg import GreedyconcatEdgeCylinderCfg
from instinct_mj.utils.noise import CropAndResizeCfg, DepthNormalizationCfg, GaussianBlurNoiseCfg

if TYPE_CHECKING:
    pass

__file_dir__ = os.path.dirname(os.path.realpath(__file__))
# NOTE: Change this to your local CASBOT parkour dataset root before training / play.
_PARKOUR_DATASET_DIR = os.path.expanduser("~/Desktop/InstinctMJ/Datasets/casbot02/parkour_motion_reference")

# Keep the original amp_mjlab meshes intact. At 7 cm, the upper-torso mesh
# exceeds MuJoCo-Warp's per-pair heightfield contact limit in fallen poses.
# 8/9 cm also overflow in stress checks; 10 cm passes sampled random poses.
# This affects collision sampling only; terrain mesh generation stays at 7 cm.
CASBOT02_PARKOUR_HFIELD_RESOLUTION = 0.10

# ---------------------------------------------------------------------------
# Motion reference configs
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class AmassMotionCfg(AmassMotionCfgBase):
    """Parkour AMASS motion buffer config for CASBOT_02."""

    path: str = _PARKOUR_DATASET_DIR
    retargetting_func: object | None = None
    filtered_motion_selection_filepath: str | None = os.path.join(
        _PARKOUR_DATASET_DIR,
        "parkour_motion_without_run.yaml",
    )
    motion_start_from_middle_range: list[float] = field(default_factory=lambda: [0.0, 0.9])
    motion_start_height_offset: float = 0.0
    ensure_link_below_zero_ground: bool = False
    buffer_device: str = "output_device"
    motion_interpolate_func: object = field(default_factory=lambda: motion_interpolate_bilinear)
    velocity_estimation_method: str = "frontward"


motion_reference_cfg = MotionReferenceManagerCfg(
    name="motion_reference",
    entity_name="robot",
    robot_model_path=CASBOT02_PARKOUR_MJCF_PATH,
    link_of_interests=[
        "torso",
        # left leg
        "leg_l2_link",  # hip roll (thigh)
        "leg_l4_link",  # knee (shank)
        "leg_l6_link",  # ankle roll (foot)
        # right leg
        "leg_r2_link",
        "leg_r4_link",
        "leg_r6_link",
        # waist
        "waist_yaw_link",
        # left arm
        "left_shoulder_roll_link",
        "left_elbow_pitch_link",
        "left_wrist_yaw_link",
        # right arm
        "right_shoulder_roll_link",
        "right_elbow_pitch_link",
        "right_wrist_yaw_link",
    ],
    # 14 links: torso(0) + L leg(1-3) + R leg(4-6) + waist(7) + L arm(8-10) + R arm(11-13)
    symmetric_augmentation_link_mapping=[0, 4, 5, 6, 1, 2, 3, 7, 11, 12, 13, 8, 9, 10],
    symmetric_augmentation_joint_mapping=list(CASBOT02_PARKOUR_SYMMETRIC_JOINT_MAPPING),
    symmetric_augmentation_joint_reverse_buf=list(CASBOT02_PARKOUR_SYMMETRIC_JOINT_REVERSE_BUF),
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    motion_buffers={"run_walk": AmassMotionCfg()},
    mp_split_method="Even",
)

# ---------------------------------------------------------------------------
# Scene config
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class Casbot02ParkourSceneCfg(SceneCfg):
    """Scene configuration for the CASBOT_02 Parkour task."""


@dataclass(kw_only=True)
class Casbot02ParkourAmpEnvCfg(InstinctLabRLEnvCfg):
    """Environment configuration for CASBOT_02 Parkour AMP."""

    scene: Casbot02ParkourSceneCfg = field(default_factory=Casbot02ParkourSceneCfg)
    decimation: int = 4
    observations: dict = field(default_factory=dict)
    actions: dict = field(default_factory=dict)
    rewards: dict = field(default_factory=lambda: {"rewards": {}})
    terminations: dict = field(default_factory=dict)
    commands: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    curriculum: dict = field(default_factory=dict)
    monitors: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        pass


def _build_casbot02_robot_entity() -> EntityCfg:
    """Build CASBOT_02 robot entity config."""
    init_state = copy.deepcopy(CASBOT02_PARKOUR_INIT_STATE)
    return EntityCfg(
        init_state=init_state,
        spec_fn=get_casbot02_parkour_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(copy.deepcopy(act) for act in casbot02_parkour_delayed_actuator_cfgs),
            soft_joint_pos_limit_factor=0.9,
        ),
        sort_actuators=True,
    )


def instinct_casbot02_parkour_amp_env_cfg(
    *,
    play: bool = False,
) -> Casbot02ParkourAmpEnvCfg:
    """Build the CASBOT_02 parkour AMP environment configuration.

    Args:
        play: If True, apply play-mode overrides (fewer envs, relaxed
            termination, etc.).

    Returns:
        A ``Casbot02ParkourAmpEnvCfg`` instance.
    """
    # Build robot entity
    robot_entity = _build_casbot02_robot_entity()

    # Build scene
    scene_cfg = Casbot02ParkourSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        entities={"robot": robot_entity},
        sensors=(),
        spec_fn=None,
    )

    cfg = Casbot02ParkourAmpEnvCfg(
        decimation=4,
        scene=scene_cfg,
        observations={},
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=copy.deepcopy(casbot02_parkour_action_scale),
                use_default_offset=True,
            ),
        },
        events={},
        seed=42,
        episode_length_s=20.0,
        rewards={"rewards": {}},
        terminations={},
        commands={},
        curriculum={},
    )

    # Basic simulation settings
    cfg.sim.mujoco.timestep = 0.005
    # The shell asset's multi-capsule feet/legs can generate >128 contacts
    # before the initial reset (199 in the paired Play startup check).
    cfg.sim.nconmax = 512
    cfg.sim.njmax = 2048
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.sim.mujoco.iterations = 10
    cfg.sim.mujoco.ls_iterations = 20
    cfg.sim.mujoco.ccd_iterations = 128

    # Viewer
    cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
    cfg.viewer.entity_name = None
    cfg.viewer.body_name = None

    # Robot init
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, 0.92)

    # -------------------------------------------------------------------
    # Terrain (reuse G1's parkour terrain definitions)
    # -------------------------------------------------------------------
    terrain_gen = copy.deepcopy(ROUGH_TERRAINS_CFG_PLAY if play else ROUGH_TERRAINS_CFG)
    terrain_gen.hfield_resolution = CASBOT02_PARKOUR_HFIELD_RESOLUTION
    edge_obstacle_cfg = GreedyconcatEdgeCylinderCfg(
        cylinder_radius=0.05,
        min_points=2,
        component_workers=0,
        merge_collinear_gap=0.09,
        merge_collinear_angle_threshold=30.0,
        merge_collinear_line_distance=0.04,
    )
    cfg.scene.terrain = InstinctTerrainImporterCfg(
        terrain_type="hacked_generator",
        terrain_generator=copy.deepcopy(terrain_gen),
        max_init_terrain_level=5,
        virtual_obstacle_source="mesh",
        virtual_obstacle_hfield_height_threshold=0.04,
        collision_debug_vis=True,
        collision_debug_rgba=(0.62, 0.2, 0.9, 0.35),
        virtual_obstacles={
            "edges": edge_obstacle_cfg,
        },
    )
    cfg.scene.spec_fn = _edit_parkour_scene_spec

    # -------------------------------------------------------------------
    # Sensors
    # -------------------------------------------------------------------
    cfg.scene.sensors = (
        ForceThresholdContactSensorCfg(
            name="contact_forces",
            primary=ContactMatch(
                mode="body",
                pattern=("leg_l6_link", "leg_r6_link"),
                entity="robot",
            ),
            fields=("force",),
            reduce="netforce",
            track_air_time=True,
            force_threshold=1.0,
            history_length=3,
        ),
        ContactSensorCfg(
            name="torso_contact_forces",
            primary=ContactMatch(mode="body", pattern="torso", entity="robot"),
            secondary=None,
            fields=("force",),
            reduce="netforce",
            track_air_time=False,
            history_length=3,
        ),
        ContactSensorCfg(
            name="undesired_contact_forces",
            primary=ContactMatch(
                mode="body",
                pattern=".*",
                entity="robot",
                exclude=("leg_l6_link", "leg_r6_link"),
            ),
            fields=("force",),
            reduce="netforce",
            track_air_time=False,
            history_length=3,
        ),
        VolumePointsCfg(
            name="leg_volume_points",
            entity_name="robot",
            body_names="leg_l6_link|leg_r6_link",
            points_generator=Grid3dPointsGeneratorCfg(
                # Match InstinctLab's CASBOT02 18DOF foot-volume sampling.
                x_min=-0.025,
                x_max=0.12,
                x_num=10,
                y_min=-0.03,
                y_max=0.03,
                y_num=5,
                z_min=-CASBOT02_FOOT_SOLE_HEIGHT - 0.005,
                z_max=-CASBOT02_FOOT_SOLE_HEIGHT + 0.035,
                z_num=2,
            ),
            debug_vis=False,
        ),
        RayCastSensorCfg(
            name="left_height_scanner",
            frame=ObjRef(type="body", name="leg_l6_link", entity="robot"),
            pattern=GridPatternCfg(resolution=0.12, size=(0.12, 0.0)),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=False,
        ),
        RayCastSensorCfg(
            name="right_height_scanner",
            frame=ObjRef(type="body", name="leg_r6_link", entity="robot"),
            pattern=GridPatternCfg(resolution=0.12, size=(0.12, 0.0)),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=False,
        ),
        NoisyGroupedRayCasterCameraCfg(
            name="camera",
            frame=ObjRef(type="body", name=CASBOT02_DEPTH_CAMERA_LINK, entity="robot"),
            pattern=PinholeCameraPatternCfg(
                width=64,
                height=36,
                fovy=CASBOT02_DEPTH_CAMERA_FOV_Y_DEG,
            ),
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(CASBOT02_DEPTH_CAMERA_FOV_X_DEG) / 2.0),
            vertical_aperture=2 * math.tan(math.radians(CASBOT02_DEPTH_CAMERA_FOV_Y_DEG) / 2.0),
            ray_alignment="yaw",
            offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
                pos=CASBOT02_DEPTH_CAMERA_OFFSET_POS,
                rot=CASBOT02_DEPTH_CAMERA_OFFSET_ROT,
                convention="world",
            ),
            data_types=["distance_to_image_plane"],
            depth_clipping_behavior="max",
            noise_pipeline={
                "crop_and_resize": CropAndResizeCfg(crop_region=CASBOT02_DEPTH_CAMERA_CROP_REGION),
                "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
                "depth_normalization": DepthNormalizationCfg(
                    depth_range=(0.0, 2.5),
                    normalize=True,
                    output_range=(0.0, 1.0),
                ),
            },
            data_histories={"distance_to_image_plane_noised": 37},
            min_distance=0.1,
            max_distance=2.5,
            debug_vis=False,
        ),
    )
    # Add motion reference sensor
    motion_reference_sensor_cfg = copy.deepcopy(motion_reference_cfg)
    existing_sensors = tuple(
        sensor_cfg for sensor_cfg in cfg.scene.sensors if sensor_cfg.name != motion_reference_sensor_cfg.name
    )
    cfg.scene.sensors = existing_sensors + (motion_reference_sensor_cfg,)

    # -------------------------------------------------------------------
    # Commands — terrain-aware velocity commands
    # -------------------------------------------------------------------
    cfg.commands = {
        "base_velocity": PoseVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(8.0, 12.0),
            debug_vis=False,
            velocity_control_stiffness=2.0,
            heading_control_stiffness=2.0,
            rel_standing_envs=0.05,
            ranges=PoseVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),
            ),
            random_velocity_terrain=["perlin_rough_stand"],
            velocity_ranges={
                "perlin_rough": {"lin_vel_x": (0.45, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
                "perlin_rough_stand": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
                "square_gaps": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
                "pyramid_stairs": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
                "pyramid_stairs_high": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
                "pyramid_stairs_inv": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
                "pyramid_stairs_inv_high": {
                    "lin_vel_x": (0.45, 0.8),
                    "lin_vel_y": (0.0, 0.0),
                    "ang_vel_z": (-1.0, 1.0),
                },
                "boxes": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
                "dense_boxes": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
                "hf_pyramid_slope_inv": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            },
            only_positive_lin_vel_x=True,
            lin_vel_threshold=0.0,
            ang_vel_threshold=0.0,
            target_dis_threshold=0.4,
        ),
    }

    # -------------------------------------------------------------------
    # Observations
    # -------------------------------------------------------------------
    policy_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            history_length=8,
            flatten_history_dim=True,
            scale=0.25,
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=8,
            flatten_history_dim=True,
        ),
        "velocity_commands": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=8,
            flatten_history_dim=True,
            noise=None,
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=".*"),
            },
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            history_length=8,
            flatten_history_dim=True,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=".*"),
            },
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
            scale=0.05,
            history_length=8,
            flatten_history_dim=True,
        ),
        "actions": ObservationTermCfg(
            func=envs_mdp.last_action,
            history_length=8,
            flatten_history_dim=True,
        ),
        "depth_image": ObservationTermCfg(
            func=envs_mdp.delayed_visualizable_image,
            params={
                "data_type": "distance_to_image_plane_noised_history",
                "sensor_cfg": SceneEntityCfg("camera"),
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "debug_vis": False,
            },
            noise=None,
        ),
    }
    critic_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.base_lin_vel,
            history_length=8,
            flatten_history_dim=True,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            history_length=8,
            flatten_history_dim=True,
            scale=0.25,
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            history_length=8,
            flatten_history_dim=True,
        ),
        "velocity_commands": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=8,
            flatten_history_dim=True,
            noise=None,
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=".*"),
            },
            history_length=8,
            flatten_history_dim=True,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=".*"),
            },
            scale=0.05,
            history_length=8,
            flatten_history_dim=True,
        ),
        "actions": ObservationTermCfg(
            func=envs_mdp.last_action,
            history_length=8,
            flatten_history_dim=True,
        ),
        "depth_image": ObservationTermCfg(
            func=envs_mdp.delayed_visualizable_image,
            params={
                "data_type": "distance_to_image_plane_noised_history",
                "sensor_cfg": SceneEntityCfg("camera"),
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "debug_vis": False,
            },
            noise=None,
        ),
    }
    cfg.observations["policy"] = ObservationGroupCfg(
        terms=policy_terms,
        concatenate_terms=False,
        enable_corruption=True,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=False,
        enable_corruption=False,
    )

    amp_policy_terms = {
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            history_length=10,
            flatten_history_dim=True,
        ),
        "joint_pos_rel": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", preserve_order=True),
            },
            history_length=10,
            flatten_history_dim=True,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", preserve_order=True),
            },
            scale=0.05,
            history_length=10,
            flatten_history_dim=True,
        ),
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
            history_length=10,
            flatten_history_dim=True,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
            history_length=10,
            flatten_history_dim=True,
        ),
    }
    amp_reference_terms = {
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity_reference_as_state,
            params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
            history_length=10,
            flatten_history_dim=True,
        ),
        "joint_pos_rel": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel_reference_as_state,
            params={
                "asset_cfg": SceneEntityCfg(name="motion_reference"),
                "robot_cfg": SceneEntityCfg(name="robot"),
            },
            history_length=10,
            flatten_history_dim=True,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel_reference_as_state,
            params={
                "asset_cfg": SceneEntityCfg(name="motion_reference"),
                "robot_cfg": SceneEntityCfg(name="robot"),
            },
            scale=0.05,
            history_length=10,
            flatten_history_dim=True,
        ),
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.base_lin_vel_reference_as_state,
            params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
            history_length=10,
            flatten_history_dim=True,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel_reference_as_state,
            params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
            history_length=10,
            flatten_history_dim=True,
        ),
    }
    cfg.observations["amp_policy"] = ObservationGroupCfg(
        terms=amp_policy_terms,
        concatenate_terms=False,
        enable_corruption=False,
    )
    cfg.observations["amp_reference"] = ObservationGroupCfg(
        terms=amp_reference_terms,
        concatenate_terms=False,
        enable_corruption=False,
    )

    # -------------------------------------------------------------------
    # Rewards (same structure as G1 parkour, adapted to CASBOT body names)
    # -------------------------------------------------------------------
    cfg.rewards = {
        "rewards": {
            # ---------- Task rewards ----------
            "track_lin_vel_xy_exp": RewardTermCfg(
                func=parkour_mdp.track_lin_vel_xy_exp,
                weight=2.0,
                params={"command_name": "base_velocity", "std": 0.5},
            ),
            "track_ang_vel_z_exp": RewardTermCfg(
                func=parkour_mdp.track_ang_vel_z_exp,
                weight=2.0,
                params={"command_name": "base_velocity", "std": 0.5},
            ),
            "heading_error": RewardTermCfg(
                func=parkour_mdp.heading_error,
                weight=-1.0,
                params={"command_name": "base_velocity"},
            ),
            "dont_wait": RewardTermCfg(
                func=parkour_mdp.dont_wait,
                weight=-0.5,
                params={"command_name": "base_velocity"},
            ),
            "is_alive": RewardTermCfg(func=envs_mdp.is_alive, weight=3.0),
            "stand_still": RewardTermCfg(
                func=parkour_mdp.stand_still,
                weight=-0.3,
                params={"command_name": "base_velocity", "offset": 4.0},
            ),
            # ---------- Regularization rewards ----------
            "volume_points_penetration": RewardTermCfg(
                func=parkour_mdp.volume_points_penetration,
                weight=-4.0,
                params={"sensor_name": "leg_volume_points"},
            ),
            "feet_air_time": RewardTermCfg(
                func=parkour_mdp.feet_air_time,
                weight=1.0,
                params={
                    "command_name": "base_velocity",
                    "sensor_name": "contact_forces",
                    "vel_threshold": 0.15,
                },
            ),
            "feet_slide": RewardTermCfg(
                func=parkour_mdp.feet_slide,
                weight=-0.4,
                params={
                    "sensor_name": "contact_forces",
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=("leg_l6_link", "leg_r6_link"),
                    ),
                    "threshold": 1.0,
                },
            ),
            "joint_deviation_hip": RewardTermCfg(
                func=parkour_mdp.joint_deviation_square,
                weight=-0.5,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=(
                            "leg_l2_joint",
                            "leg_l3_joint",
                            "leg_r2_joint",
                            "leg_r3_joint",
                        ),
                    )
                },
            ),
            "ang_vel_xy_l2": RewardTermCfg(func=parkour_mdp.ang_vel_xy_l2, weight=-0.05),
            "dof_torques_l2": RewardTermCfg(
                func=parkour_mdp.joint_torques_l2,
                weight=-1.5e-7,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=(
                            "leg_l1_joint",
                            "leg_l2_joint",
                            "leg_l3_joint",
                            "leg_l4_joint",
                            "leg_l5_joint",
                            "leg_l6_joint",
                            "leg_r1_joint",
                            "leg_r2_joint",
                            "leg_r3_joint",
                            "leg_r4_joint",
                            "leg_r5_joint",
                            "leg_r6_joint",
                        ),
                    )
                },
            ),
            "dof_acc_l2": RewardTermCfg(
                func=envs_mdp.joint_acc_l2,
                weight=-1.25e-7,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
            ),
            "dof_vel_l2": RewardTermCfg(
                func=envs_mdp.joint_vel_l2,
                weight=-1e-4,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
            ),
            "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.005),
            "flat_orientation_l2": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=-3.0),
            "pelvis_orientation_l2": RewardTermCfg(
                func=parkour_mdp.link_orientation,
                weight=-3.0,
                params={"asset_cfg": SceneEntityCfg("robot", body_names="torso")},
            ),
            "feet_flat_ori": RewardTermCfg(
                func=parkour_mdp.feet_orientation_contact,
                weight=-0.4,
                params={
                    "sensor_name": "contact_forces",
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=("leg_l6_link", "leg_r6_link"),
                    ),
                },
            ),
            "feet_at_plane": RewardTermCfg(
                func=parkour_mdp.feet_at_plane,
                weight=-0.1,
                params={
                    "contact_sensor_name": "contact_forces",
                    "left_height_scanner_name": "left_height_scanner",
                    "right_height_scanner_name": "right_height_scanner",
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=("leg_l6_link", "leg_r6_link"),
                    ),
                    "height_offset": CASBOT02_FOOT_SOLE_HEIGHT,
                },
            ),
            "feet_close_xy": RewardTermCfg(
                func=parkour_mdp.feet_close_xy_gauss,
                weight=0.4,
                params={
                    "threshold": 0.12,
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=("leg_l6_link", "leg_r6_link"),
                    ),
                    "std": math.sqrt(0.05),
                },
            ),
            "energy": RewardTermCfg(
                func=parkour_mdp.motors_power_square,
                weight=-5e-5,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=(
                            "leg_l1_joint",
                            "leg_l2_joint",
                            "leg_l3_joint",
                            "leg_l4_joint",
                            "leg_l5_joint",
                            "leg_l6_joint",
                            "leg_r1_joint",
                            "leg_r2_joint",
                            "leg_r3_joint",
                            "leg_r4_joint",
                            "leg_r5_joint",
                            "leg_r6_joint",
                        ),
                    ),
                    "normalize_by_stiffness": True,
                },
            ),
            "freeze_upper_body": RewardTermCfg(
                func=parkour_mdp.joint_deviation_l1,
                weight=-0.004,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=tuple(name for name in CASBOT02_PARKOUR_JOINT_NAMES if name.startswith("upper_")),
                    )
                },
            ),
            # ---------- Safety rewards ----------
            "dof_pos_limits": RewardTermCfg(
                func=envs_mdp.joint_pos_limits,
                weight=-1.0,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
            ),
            "torque_limits": RewardTermCfg(
                func=parkour_mdp.applied_torque_limits_by_ratio,
                weight=-0.01,
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                    "limit_ratio": 0.8,
                },
            ),
            "undesired_contacts": RewardTermCfg(
                func=parkour_mdp.undesired_contacts,
                weight=-1.0,
                params={"sensor_name": "undesired_contact_forces", "threshold": 1.0},
            ),
        },
    }

    # -------------------------------------------------------------------
    # Curriculum
    # -------------------------------------------------------------------
    cfg.curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=parkour_mdp.tracking_exp_vel,
            params={
                "lin_vel_threshold": (0.3, 0.6),
                "ang_vel_threshold": (0.0, 0.0),
            },
        ),
    }

    # -------------------------------------------------------------------
    # Terminations
    # -------------------------------------------------------------------
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "terrain_out_bound": TerminationTermCfg(
            func=envs_mdp.terrain_out_of_bounds,
            time_out=True,
            params={"distance_buffer": 2.0},
        ),
        "base_contact": TerminationTermCfg(
            func=parkour_mdp.illegal_contact,
            params={"sensor_name": "torso_contact_forces", "threshold": 1.0},
        ),
        "bad_orientation": TerminationTermCfg(
            func=envs_mdp.bad_orientation,
            params={"limit_angle": 1.0},
        ),
        "root_height": TerminationTermCfg(
            func=parkour_mdp.root_height_below_env_origin_minimum,
            params={"minimum_height": 0.5},
        ),
        "dataset_exhausted": TerminationTermCfg(
            func=envs_mdp.dataset_exhausted,
            time_out=True,
            params={
                "reference_cfg": SceneEntityCfg(name="motion_reference"),
                "print_reason": False,
                "reset_without_notice": True,
            },
        ),
    }

    # -------------------------------------------------------------------
    # Events (domain randomization)
    # -------------------------------------------------------------------
    cfg.events = {
        "physics_material": EventTermCfg(
            # Use the decorated mjlab term so friction is expanded per world,
            # rather than writing different samples into a shared model field.
            func=envs_mdp.dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
                "ranges": (0.3, 1.6),
                "operation": "abs",
                "shared_random": True,
            },
        ),
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
                "velocity_range": {
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2),
                    "z": (-0.2, 0.2),
                    "roll": (-0.2, 0.2),
                    "pitch": (-0.2, 0.2),
                    "yaw": (-0.2, 0.2),
                },
            },
        ),
        "register_virtual_obstacles": EventTermCfg(
            func=envs_mdp.register_virtual_obstacle_to_sensor,
            mode="startup",
            params={
                "sensor_cfgs": SceneEntityCfg("leg_volume_points"),
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.15, 0.15),
                "velocity_range": (0.0, 0.0),
            },
        ),
    }

    if not play:
        # Match HANDOFF's mass/COM/gain ranges; include arms in PD randomization.
        # body_mass intentionally changes mass only, as in that validated setup.
        # COM ranges come from HANDOFF's inherited make_velocity_env_cfg.
        cfg.events.update(
            {
                "base_mass": EventTermCfg(
                    mode="startup",
                    func=envs_mdp.dr.body_mass,
                    params={
                        "asset_cfg": SceneEntityCfg("robot", body_names=("torso",)),
                        "ranges": (-4.0, 4.0),
                        "operation": "add",
                        "distribution": "uniform",
                    },
                ),
                "non_base_mass": EventTermCfg(
                    mode="startup",
                    func=envs_mdp.dr.body_mass,
                    params={
                        "asset_cfg": SceneEntityCfg("robot", body_names=(r"^(?!torso$).+$",)),
                        "ranges": (0.8, 1.2),
                        "operation": "scale",
                        "distribution": "uniform",
                        "shared_random": False,
                    },
                ),
                "base_com": EventTermCfg(
                    mode="startup",
                    func=envs_mdp.dr.body_com_offset,
                    params={
                        "asset_cfg": SceneEntityCfg("robot", body_names=("torso",)),
                        "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
                        "operation": "add",
                        "distribution": "uniform",
                    },
                ),
                "actuator_gains": EventTermCfg(
                    mode="startup",
                    func=envs_mdp.dr.pd_gains,
                    params={
                        # LEG_HEAVY, LEG_LIGHT, ARM_HEAVY: 12 leg + 6 arm joints.
                        "asset_cfg": SceneEntityCfg("robot", actuator_ids=[0, 1, 2]),
                        "kp_range": (0.85, 1.15),
                        "kd_range": (0.85, 1.15),
                        "operation": "scale",
                        "distribution": "uniform",
                    },
                ),
                "camera_installation": EventTermCfg(
                    mode="startup",
                    func=envs_mdp.randomize_camera_offsets,
                    params={
                        "asset_cfg": SceneEntityCfg("camera"),
                        "offset_pose_ranges": {
                            "x": (-0.03, 0.03),
                            "y": (-0.03, 0.03),
                            "z": (-0.03, 0.03),
                            "roll": (-math.radians(3), math.radians(3)),
                            "pitch": (-math.radians(3), math.radians(3)),
                            "yaw": (-math.radians(3), math.radians(3)),
                        },
                        "distribution": "uniform",
                    },
                ),
            }
        )

    # -------------------------------------------------------------------
    # Play-mode overrides
    # -------------------------------------------------------------------
    if play:
        cfg.scene.num_envs = 10
        cfg.scene.env_spacing = 2.5
        cfg.episode_length_s = 10.0
        cfg.scene.terrain.terrain_generator.num_rows = 4
        cfg.scene.terrain.terrain_generator.num_cols = 10

        leg_volume_points_sensor = next(
            sensor_cfg for sensor_cfg in cfg.scene.sensors if sensor_cfg.name == "leg_volume_points"
        )
        leg_volume_points_sensor.debug_vis = True

        cfg.scene.terrain.collision_debug_vis = False
        cfg.commands["base_velocity"].debug_vis = True
        cfg.commands["base_velocity"].patch_vis = False
        cfg.terminations["root_height"] = None
        cfg.events["physics_material"] = None
        cfg.events["reset_robot_joints"].params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }

    return cfg


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------


def instinct_casbot02_parkour_amp_final_cfg(
    *,
    play: bool = False,
) -> Casbot02ParkourAmpEnvCfg:
    """Create the final CASBOT_02 parkour AMP env configuration.

    Args:
        play: If True, apply play-mode overrides.

    Returns:
        A fully-built ``Casbot02ParkourAmpEnvCfg`` instance.
    """
    cfg = instinct_casbot02_parkour_amp_env_cfg(play=play)

    if play:
        from mjlab.viewer.viewer_config import ViewerConfig

        cfg.viewer = ViewerConfig(
            lookat=(0.0, 0.75, 0.0),
            distance=4.123105625617661,
            elevation=-14.036243467926479,
            azimuth=180.0,
            origin_type=ViewerConfig.OriginType.WORLD,
            entity_name=None,
        )
        cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
        cfg.viewer.entity_name = None
        cfg.viewer.body_name = None

    return cfg
