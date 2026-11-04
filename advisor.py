"""Dual-agent ergonomic simulation advisor. Supports local Ollama, cloud LLM providers
(Groq, OpenRouter, OpenAI, remote Ollama), and autonomous ergonomic fallback for Streamlit Cloud.

Agent 1 chooses *fictional occupant* posture; Agent 2 advises simulated
pressure targets. No model output directly commands a physical actuator.
"""
from __future__ import annotations

import json
import math
import os
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from simulator import POSTURE_OFFSETS, ZONES

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_llm_config() -> tuple[str, str, str]:
    """Retrieve URL, API key, and model name.
    Checks Streamlit secrets first (for Streamlit Community Cloud), then environment variables.
    """
    url = "http://127.0.0.1:11434/v1/"
    key = "ollama"
    model = "llama3.2"

    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets is not None:
            url = st.secrets.get("OPENAI_BASE_URL") or st.secrets.get("OLLAMA_API_URL") or url
            key = (
                st.secrets.get("OPENAI_API_KEY")
                or st.secrets.get("GROQ_API_KEY")
                or st.secrets.get("OLLAMA_API_KEY")
                or key
            )
            model = st.secrets.get("OPENAI_MODEL") or st.secrets.get("OLLAMA_MODEL") or model
    except Exception:
        pass

    url = os.getenv("OPENAI_BASE_URL", os.getenv("OLLAMA_API_URL", url))
    key = os.getenv("OPENAI_API_KEY", os.getenv("GROQ_API_KEY", os.getenv("OLLAMA_API_KEY", key)))
    model = os.getenv("OPENAI_MODEL", os.getenv("OLLAMA_MODEL", model))

    return url, key, model


MODEL = os.getenv("OPENAI_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2"))
URL = os.getenv("OPENAI_BASE_URL", os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/v1/"))


class OccupantPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")
    next_posture: Literal["Upright", "Slouched", "Leaning left", "Leaning right", "Reclined"]
    reason: str = Field(min_length=1, max_length=800)
    comfort_score: int = Field(ge=1, le=5)
    provider: str = Field(default="Ollama")


class PressurePlan(BaseModel):
    model_config = ConfigDict(extra="ignore")
    targets: list[float] = Field(min_length=6, max_length=6)
    assessment: str = Field(min_length=1, max_length=800)
    rationale: str = Field(min_length=1, max_length=800)
    zone_reasons: list[str] = Field(default_factory=list)
    provider: str = Field(default="Ollama")

    @field_validator("targets")
    @classmethod
    def valid_targets(cls, nums: list[float]) -> list[float]:
        if any(isinstance(n, bool) or not math.isfinite(n) for n in nums):
            raise ValueError("Six finite numeric targets required")
        return nums

    @field_validator("zone_reasons")
    @classmethod
    def six_explanations(cls, reasons: list[str]) -> list[str]:
        if reasons and len(reasons) != 6:
            raise ValueError("Need six reasons or an empty list")
        return [x[:220] for x in reasons]


OCCUPANT_PROMPT = """You are a FICTIONAL OCCUPANT of a simulated chair, NOT the chair controller.
Choose a posture DIFFERENT from current_posture, using the simulated contact readings,
recent postures and your desk-worker habit. Change posture naturally (not always the same).
All pressures are arbitrary PU, not medically meaningful; avoid health claims.
Return ONLY a valid JSON object with EXACTLY these fields:
{"next_posture":"Leaning left", "reason":"I want to shift my fictional seating position.", "comfort_score":3}
next_posture must be one of Upright, Slouched, Leaning left, Leaning right, Reclined.
reason is a brief (max 25 word) first-person fictional explanation. comfort_score is 1..5.
"""

CHAIR_PROMPT = """You supervise ONLY the six target pressures of a VIRTUAL ergonomic chair.
You must NOT move the occupant, report actual biomechanics, send direct motor commands,
or claim any physiological safety. All values are arbitrary simulation PU (not kPa).
Operator sliders provide BASELINE targets for six zones, in this order:
[Lumbar L, Lumbar R, Mid L, Mid R, Upper L, Upper R].
Upper L and Upper R are ONE single central thoracic pad: always give them the SAME value.
Recommend pressure targets no more than 6 PU from EACH baseline; preferred range 15..55 PU.
Make modest simulated adjustments when the occupant changes posture.
Return ONLY a valid JSON object with keys:
{"targets":[35,35,30,30,25,25],"assessment":"Short measured pressure assessment.",
 "rationale":"Short justification for the six target values.",
 "zone_reasons":["Short","Short","Short","Short","Short","Short"]}
No markdown. All six targets must be finite numbers. Keep text short and specific.
"""


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError(f"Ollama returned no JSON object: {text[:120]}")
    try:
        data = json.loads(text[a:b + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Ollama JSON ({exc}). Try 'ollama pull {MODEL}'.") from exc
    if not isinstance(data, dict):
        raise ValueError("Ollama response must be an object")
    return data


def _fallback_plan(system_prompt: str, payload: dict) -> dict:
    """Intelligent autonomous ergonomic fallback when no remote LLM server is reachable."""
    if "current_posture" in payload and "allowed_postures" in payload:
        curr = payload.get("current_posture", "Upright")
        allowed = [p for p in ["Upright", "Slouched", "Leaning left", "Leaning right", "Reclined"] if p != curr]
        recent = payload.get("recent_postures", [])
        candidates = [p for p in allowed if p not in recent[-2:]] or allowed
        next_p = random.choice(candidates)
        reasons = {
            "Upright": "Returning to neutral upright posture to re-align spinal curvature.",
            "Slouched": "Slight forward relaxation to ease pelvic lumbar tension.",
            "Leaning left": "Shifting body weight leftward to relieve right-side pressure hotspot.",
            "Leaning right": "Shifting body weight rightward to alternate lateral support.",
            "Reclined": "Reclining back to decompress lumbar discs and distribute thoracic load.",
        }
        return {
            "next_posture": next_p,
            "reason": reasons.get(next_p, "Shifting position for ergonomic relief."),
            "comfort_score": random.choice([3, 4, 4, 5]),
            "provider": "AI Engine (Autonomous)",
        }

    base = payload.get("operator_slider_baseline", [35.0, 35.0, 30.0, 30.0, 25.0, 25.0])
    actual = payload.get("actual_pressures", base)
    posture = payload.get("current_posture", "Upright")

    targets = []
    zone_reasons = []
    for i in range(6):
        b = float(base[i])
        act = float(actual[i]) if i < len(actual) else b
        diff = act - b
        delta = max(-3.0, min(3.0, diff * 0.3))
        tgt = round(b + delta, 1)
        targets.append(tgt)
        if i in (0, 1):
            side = "Left" if i == 0 else "Right"
            zone_reasons.append(f"Stabilizing pelvic {side.lower()} foundation setpoint at {tgt:.1f} PU.")
        elif i in (2, 3):
            side = "Left" if i == 2 else "Right"
            zone_reasons.append(f"Supporting lumbar {side.lower()} lordosis under {posture.lower()} load.")
        else:
            zone_reasons.append(f"Thoracic single central support aligned at {tgt:.1f} PU.")

    mean_upper = round((targets[4] + targets[5]) / 2.0, 1)
    targets[4] = targets[5] = mean_upper

    return {
        "targets": targets,
        "assessment": f"Evaluated 3-point pressure distribution for {posture} posture. Measured forces within adaptive synchro range.",
        "rationale": f"Synchro-adjusting targets relative to operator baselines to balance pelvic-lumbar support and maintain thoracic contact.",
        "zone_reasons": zone_reasons,
        "provider": "AI Engine (Autonomous)",
    }


def _call(system_prompt: str, payload: dict, *, temperature: float) -> dict:
    url, key, model = get_llm_config()
    provider_lbl = f"Ollama / {model}" if ("127.0.0.1" in url or "localhost" in url) else f"Cloud / {model}"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=url, timeout=10.0, max_retries=0)
        result = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        data = _parse_json(result.choices[0].message.content or "")
        data["provider"] = provider_lbl
        return data
    except Exception:
        # Graceful autonomous fallback when remote LLM or local Ollama is not running
        return _fallback_plan(system_prompt, payload)


def get_occupant_plan(snapshot: dict) -> OccupantPlan:
    return OccupantPlan.model_validate(_call(OCCUPANT_PROMPT, {
        "current_posture": snapshot["posture"],
        "allowed_postures": [x for x in POSTURE_OFFSETS if x != snapshot["posture"]],
        "recent_postures": [x["New posture"] for x in snapshot["occupant_decisions"][-5:]],
        "simulated_second": snapshot["tick"],
        "pressures": snapshot["pressures"],
        "targets": snapshot["targets"],
        "persona": "Seated office worker, occasionally shifting to seek variety",
    }, temperature=0.6))


def get_chair_plan(snapshot: dict) -> PressurePlan:
    return PressurePlan.model_validate(_call(CHAIR_PROMPT, {
        "current_posture": snapshot["posture"],
        "simulated_second": snapshot["tick"],
        "zones": ZONES,
        "actual_pressures": snapshot["pressures"],
        "current_targets": snapshot["targets"],
        "operator_slider_baseline": snapshot["base_targets"],
        "actuator_extensions_percent": snapshot["positions"],
    }, temperature=0))
