"""Register Instinct Mj parkour CASBOT_02 tasks."""

from instinct_mj.tasks.registry import register_instinct_task

from .agents.instinct_rl_amp_cfg import Casbot02ParkourPPORunnerCfg


def _parkour_amp_env_cfg(play: bool):
    from .casbot02_parkour_target_amp_cfg import instinct_casbot02_parkour_amp_final_cfg

    return instinct_casbot02_parkour_amp_final_cfg(play=play)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-CASBOT02-v0",
    env_cfg_factory=lambda: _parkour_amp_env_cfg(play=False),
    play_env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True),
    instinct_rl_cfg_factory=Casbot02ParkourPPORunnerCfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-CASBOT02-Play-v0",
    env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True),
    play_env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True),
    instinct_rl_cfg_factory=Casbot02ParkourPPORunnerCfg,
)
