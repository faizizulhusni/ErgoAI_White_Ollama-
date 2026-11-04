"""Virtual ergonomic chair. All pressure values are artificial simulation units (PU).

NOT a clinically calibrated device, a validated actuator model, or a human-use controller.
"""
from __future__ import annotations

import math
import random
from typing import Any

ZONES = [
    "Lumbar L", "Lumbar R", "Mid L", "Mid R", "Upper L", "Upper R"
]
ROWS = ["Lumbar", "Mid", "Upper"]
DEFAULT_TARGETS = [35.0, 35.0, 30.0, 30.0, 25.0, 25.0]
TARGET_MIN_PU = 15.0        # Demo numbers only; NOT safe human-use limits
TARGET_MAX_PU = 55.0
TRIP_PRESSURE_PU = 70.0     # Artificial test threshold, NOT a medical limit
MAX_POSITION_PCT = 70.0
MAX_STEP_PCT = 4.0          # Max simulated actuator movement per 1s tick
MAX_AI_TARGET_CHANGE_PU = 8.0
SEAT_OCCUPANCY_THRESHOLD = 20.0
PROPORTIONAL_GAIN = 0.65
# Physical chair structure: zone index pairs that are ONE physical pad/actuator.
# The upper (thoracic) support is a single central pad with no left/right split,
# so "Upper L" and "Upper R" always share one position, pressure and target.
LINKED_ZONES = [(4, 5)]


def _link(values: list[float], ndigits: int = 2) -> list[float]:
    """Collapse linked zone pairs to their shared (mean) value."""
    out = list(values)
    for a, b in LINKED_ZONES:
        shared = round((float(out[a]) + float(out[b])) / 2.0, ndigits)
        out[a] = out[b] = shared
    return out

POSTURE_OFFSETS = {
    "Upright": [0, 0, 0, 0, 0, 0],
    "Slouched": [-5, -5, 3, 3, 8, 8],
    "Leaning left": [9, -7, 8, -7, 7, -6],
    "Leaning right": [-7, 9, -7, 8, -6, 7],
    "Reclined": [4, 4, 2, 2, 1, 1],
}


def new_state() -> dict[str, Any]:
    state = {
        "seat_present": True, "posture": "Upright", "load": 1.0,
        "fault": False, "emergency": False,
        "positions": [0.0] * 6,
        "targets": DEFAULT_TARGETS.copy(),
        "base_targets": DEFAULT_TARGETS.copy(),
        "pressures": [0.0] * 6,
        "seat_signal": 0.0,
        "tick": 0,
        "status": "Ready",
        "history": [],
        "chair_decisions": [],
        "occupant_decisions": [],
    }
    state["seat_signal"] = 75.0
    state["pressures"] = sense_pressures(state, noise=False)
    return state


def occupied(state: dict) -> bool:
    return float(state["seat_signal"]) >= SEAT_OCCUPANCY_THRESHOLD


def sense_pressures(state: dict, *, noise: bool = True) -> list[float]:
    """Placeholder digital twin, not an empirical human-back pressure model."""
    if not occupied(state):
        return [0.0] * 6
    offset = POSTURE_OFFSETS[state["posture"]]
    load = float(state["load"])
    answer = []
    for i, position in enumerate(state["positions"]):
        measurement_noise = random.uniform(-0.35, 0.35) if noise else 0.0
        value = (8.0 + offset[i] + 0.75 * position) * load + measurement_noise
        answer.append(round(max(0.0, value), 2))
    return _link(answer)


def safe_plan(state: dict, proposal: list[float]) -> tuple[list[float], list[str]]:
    """Validate an LLM recommendation and limit its setpoint change per approval."""
    if not occupied(state) or state["emergency"] or state["fault"]:
        raise ValueError("AI plans are blocked while empty, faulted or E-stopped.")
    if not isinstance(proposal, list) or len(proposal) != len(ZONES):
        raise ValueError("Plan must provide exactly six numeric pressure targets.")
    safe = []
    notes = []
    for i, raw in enumerate(proposal):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"Invalid numeric target for {ZONES[i]}.")
    # A single physical pad can only have one target: average linked proposals.
    proposal = _link([float(x) for x in proposal], ndigits=3)
    for i, raw in enumerate(proposal):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"Invalid numeric target for {ZONES[i]}.")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite target for {ZONES[i]}.")
        base = state.get("base_targets", DEFAULT_TARGETS)[i]
        # Simulated LLM supervisor may only trim the user's slider targets by ±6 PU.
        # These are engineering *demo constraints*, NOT physical/medical safety limits.
        hard_clamped = min(TARGET_MAX_PU, max(TARGET_MIN_PU, number))
        hard_clamped = min(base + 6.0, max(base - 6.0, hard_clamped))
        old = state["targets"][i]
        limited = min(old + MAX_AI_TARGET_CHANGE_PU, max(old - MAX_AI_TARGET_CHANGE_PU, hard_clamped))
        limited = round(limited, 1)
        if limited != round(number, 1):
            notes.append(f"{ZONES[i]}: requested {number:.1f}, limited to {limited:.1f} PU")
        safe.append(limited)
    return safe, notes


def step(state: dict[str, Any]) -> dict[str, Any]:
    """Advance one simulated second, with an occupancy gate and E-stop latch."""
    state["tick"] += 1
    state["seat_signal"] = 75.0 * state["load"] if state["seat_present"] else 0.0

    if state["fault"]:
        state["emergency"] = True
        state["status"] = "SENSOR FAULT — E-STOP LATCHED"

    if not occupied(state):
        state["positions"] = [0.0] * 6
        state["pressures"] = [0.0] * 6
        state["status"] = "EMPTY — actuators neutral" if not state["emergency"] else "E-STOP LATCHED"
    elif state["emergency"]:
        state["positions"] = [0.0] * 6
        state["pressures"] = sense_pressures(state)
        state["status"] = "E-STOP LATCHED — no commands"
    else:
        previous = sense_pressures(state)
        if max(previous) >= TRIP_PRESSURE_PU:
            state["emergency"] = True
        else:
            updated = []
            for position, measured, setpoint in zip(state["positions"], previous, state["targets"]):
                movement = max(-MAX_STEP_PCT, min(MAX_STEP_PCT, PROPORTIONAL_GAIN * (setpoint - measured)))
                updated.append(round(max(0.0, min(MAX_POSITION_PCT, position + movement)), 2))
            state["positions"] = _link(updated)
            next_pressures = sense_pressures(state)
            if max(next_pressures) >= TRIP_PRESSURE_PU:
                state["emergency"] = True
            else:
                state["pressures"] = next_pressures
                state["status"] = "CONTROLLING — closed-loop regulation"
        if state["emergency"]:
            state["positions"] = [0.0] * 6
            state["pressures"] = sense_pressures(state)
            state["status"] = "PRESSURE TRIP — E-STOP LATCHED"

    record = {
        "t (s)": state["tick"],
        "Mean actual (PU)": round(sum(state["pressures"]) / 6, 2),
        "Mean target (PU)": round(sum(state["targets"]) / 6, 2) if occupied(state) else 0.0,
        "Mean extension (%)": round(sum(state["positions"]) / 6, 2),
        "Posture": state["posture"],
    }
    state["history"].append(record)
    state["history"] = state["history"][-300:]
    return state


def emergency_stop(state: dict) -> None:
    state["emergency"] = True
    state["positions"] = [0.0] * 6
    state["pressures"] = sense_pressures(state)
    state["status"] = "E-STOP LATCHED — manual reset required"


def reset_emergency(state: dict) -> bool:
    if state["fault"]:
        return False
    state["emergency"] = False
    state["status"] = "Reset accepted"
    return True