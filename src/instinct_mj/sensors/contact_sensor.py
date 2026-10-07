from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.sensor import ContactSensor, ContactSensorCfg


class ForceThresholdContactSensor(ContactSensor):
    """mjlab contact sensor with InstinctLab force-threshold air-time semantics."""

    cfg: ForceThresholdContactSensorCfg

    def update(self, dt: float) -> None:
        # mjlab 1.5.0 calls the air-time hook without dt. Keep the exact
        # substep duration instead of differencing the float32 simulation clock.
        self._air_time_dt = dt
        super().update(dt)

    def _update_air_time_tracking(self, dt: float | None = None) -> None:
        assert self._air_time_state is not None

        contact_data = self._extract_sensor_data()
        assert contact_data.force is not None

        elapsed_time = self._air_time_dt if dt is None else dt

        is_contact = torch.linalg.vector_norm(contact_data.force, dim=-1) > self.cfg.force_threshold

        state = self._air_time_state
        is_first_contact = (state.current_air_time > 0) & is_contact
        is_first_detached = (state.current_contact_time > 0) & ~is_contact

        state.last_air_time[:] = torch.where(
            is_first_contact,
            state.current_air_time + elapsed_time,
            state.last_air_time,
        )
        state.current_air_time[:] = torch.where(
            ~is_contact,
            state.current_air_time + elapsed_time,
            torch.zeros_like(state.current_air_time),
        )

        state.last_contact_time[:] = torch.where(
            is_first_detached,
            state.current_contact_time + elapsed_time,
            state.last_contact_time,
        )
        state.current_contact_time[:] = torch.where(
            is_contact,
            state.current_contact_time + elapsed_time,
            torch.zeros_like(state.current_contact_time),
        )


@dataclass(kw_only=True)
class ForceThresholdContactSensorCfg(ContactSensorCfg):
    """Contact sensor config with a force threshold for air/contact timing."""

    class_type: type = ForceThresholdContactSensor

    force_threshold: float = 1.0
    """Net contact-force threshold in newtons used only for air/contact timing."""

    def build(self) -> ForceThresholdContactSensor:
        return ForceThresholdContactSensor(self)
