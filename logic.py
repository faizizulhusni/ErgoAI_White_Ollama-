"""Deterministic application, audits, and event scheduling of LLM proposals."""
from __future__ import annotations

from simulator import POSTURE_OFFSETS, ZONES, occupied, safe_plan, sense_pressures

MAX_RECORDS = 250


def _log(state: dict, key: str, entry: dict) -> None:
    state.setdefault(key, []).append(entry)
    state[key] = state[key][-MAX_RECORDS:]


def apply_chair_proposal(
    state: dict, proposed: list[float], *, provider: str, rationale: str,
    assessment: str = "", zone_reasons: list[str] | None = None,
) -> dict:
    """Only 'safe_plan' writes accepted targets; preserve a full audit snapshot."""
    old = state["targets"].copy()
    measured = state["pressures"].copy()
    accepted, limitations = safe_plan(state, proposed)
    reasons = zone_reasons if zone_reasons and len(zone_reasons) == 6 else [""] * 6
    zones = []
    for i, zone in enumerate(ZONES):
        delta = round(accepted[i] - old[i], 1)
        clamp = abs(accepted[i] - float(proposed[i])) > 0.051
        zones.append({
            "Zone": zone,
            "Measured (PU)": round(float(measured[i]), 2),
            "Previous target (PU)": round(old[i], 1),
            "Proposed target (PU)": round(float(proposed[i]), 1),
            "Accepted target (PU)": accepted[i],
            "Change (PU)": delta,
            "Safety": "Limited by demo guardrail" if clamp else "Accepted by demo guardrail",
            "Explanation": reasons[i] if reasons[i] else "No per-zone explanation supplied by model",
        })
    result = {
        "Second": state["tick"],
        "Source": provider,
        "Posture": state["posture"],
        "Measured lumbar avg (PU)": round(sum(measured[:2]) / 2, 2),
        "Previous lumbar target avg (PU)": round(sum(old[:2]) / 2, 1),
        "Accepted lumbar target avg (PU)": round(sum(accepted[:2]) / 2, 1),
        "Lumbar target change (PU)": round((sum(accepted[:2]) - sum(old[:2])) / 2, 1),
        "Assessment": assessment.strip() or "No separate LLM assessment supplied",
        "Justification": rationale,
        "Guardrail result": "LIMITED: " + "; ".join(limitations) if limitations else "Accepted under simulation guardrails",
        "zones": zones,
    }
    state["targets"] = accepted
    _log(state, "chair_decisions", result)
    return result


def apply_occupant_proposal(
    state: dict, next_posture: str, *, provider: str, reason: str,
    comfort_score: int,
) -> dict:
    """Occupant may change ONLY the virtual posture scenario, not the actuators."""
    if not occupied(state) or state["fault"] or state["emergency"]:
        raise ValueError("Occupant position change is blocked by occupancy or safety interlock.")
    if next_posture not in POSTURE_OFFSETS:
        raise ValueError("Unknown posture; allowed values are predefined discrete postures.")
    if next_posture == state["posture"]:
        raise ValueError("Posture must actually change before recording an event.")
    if isinstance(comfort_score, bool) or not isinstance(comfort_score, int) or not (1 <= comfort_score <= 5):
        raise ValueError("Comfort score must be an integer 1 to 5.")
    before = state["posture"]
    state["posture"] = next_posture
    state["pressures"] = sense_pressures(state, noise=False)
    entry = {
        "Second": state["tick"],
        "Source": provider,
        "Previous posture": before,
        "New posture": next_posture,
        "Fictional comfort (1-5)": comfort_score,
        "Reason": reason,
    }
    _log(state, "occupant_decisions", entry)
    return entry


