"""ErgoAI Compact — Dual-agent Ollama simulation in kPa,
3-Point Sensor & Actuator Touchpoint, Live Animations & Graphs at Top, and AI Reasoning Showcase.

Run: python -m streamlit run app.py
Simulation only. Internal PU values are converted to kPa (1 PU = 0.25 kPa).
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from html import escape
import os
import time

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from advisor import MODEL, get_chair_plan, get_occupant_plan
from logic import apply_chair_proposal, apply_occupant_proposal
from simulator import (
    MAX_POSITION_PCT, MAX_STEP_PCT, PROPORTIONAL_GAIN,
    SEAT_OCCUPANCY_THRESHOLD, emergency_stop, new_state, occupied,
    reset_emergency, step,
)
from visuals import SIDE, posture_illustration

load_dotenv()
SHIFT_SECONDS = max(8, int(os.getenv("POSTURE_INTERVAL_SECONDS", "15")))

# Fixed kPa conversion (1 simulated PU = 0.25 kPa)
KPA_FACTOR = 0.25
KPA_DECIMALS = 2
MAX_STROKE_MM = 50.0  # Physical actuator stroke corresponding to 70% max extension

st.set_page_config(
    page_title="ErgoAI | Adaptive Synchro ErgoChair", page_icon="🪑",
    layout="wide", initial_sidebar_state="collapsed",
)

st.markdown("""<style>
:root{color-scheme:light!important}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp{
  background:#ffffff!important;color:#1b2c42!important;
}
[data-testid="stHeader"]{height:0!important;min-height:0!important;background:transparent!important}
[data-testid="stDecoration"], footer, #MainMenu{display:none!important}
.block-container, [data-testid="stMainBlockContainer"]{
  padding:6px 14px 8px!important;max-width:1620px!important;
}
[data-testid="stVerticalBlock"]{gap:.28rem!important}
[data-testid="stHorizontalBlock"]{gap:.5rem!important;align-items:stretch}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{min-width:0}
[data-testid="stSlider"]{padding:0 .15rem!important;margin-bottom:-4px!important}
[data-testid="stSlider"] label p{
  font-size:11.5px!important;font-weight:700!important;color:#1e354d!important;
  white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important;
}
[data-testid="stSlider"] [data-testid="stTickBar"]{display:none!important}
[data-testid="stSlider"] [data-baseweb="slider"]{padding-top:2px!important;padding-bottom:2px!important}
[data-testid="stCheckbox"]{margin-bottom:0!important;padding-top:0!important}
[data-testid="stCheckbox"] label p{font-size:11px!important;font-weight:650!important;color:#2b435c!important}
[data-testid="stButton"] button {
  min-height:28px!important;height:28px!important;padding:.15rem .45rem!important;font-size:11px!important;
  font-weight:750!important;border-radius:7px!important;border-color:#cbd7e4!important;
  background:#fff!important;color:#203550!important;
}
[data-testid="stButton"] button:hover{border-color:#2f83b2!important;color:#125b85!important}
[data-testid="stButton"] button[kind="primary"]{
  background:#d84343!important;color:white!important;border-color:#d84343!important
}
[data-testid="stPlotlyChart"]{
  width:100%!important;border:1px solid #dce5ee;
  border-radius:9px;background:#fff!important;overflow:hidden;margin-bottom:2px!important;
}
[data-testid="stMarkdownContainer"] p{margin-bottom:0!important}
*{box-sizing:border-box}

/* Header & Status Bar */
.ergo-header{display:flex;align-items:center;gap:10px;flex-wrap: nowrap;min-height:28px}
.ergo-logo{font-size:21px;letter-spacing:-.7px;font-weight:850;color:#10253a;line-height:1;white-space:nowrap}
.ergo-logo em{font-style:normal;color:#168b89}
.ergo-sub{font-size:11.5px;color:#4f667d;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.status-chip{border-radius:999px;padding:2px 8px;font-size:10px;font-weight:800;white-space:nowrap}
.chip-run{color:#147a52;background:#e6f7ef;border:1px solid #b4e6ce}
.chip-pause{color:#9c6514;background:#fff4dc;border:1px solid #f2d493}
.chip-stop{color:#b42b2b;background:#feecec;border:1px solid #f3bcbc}
.header-telemetry{margin-left:auto;font-size:10.5px;color:#526980;white-space:nowrap;display:flex;gap:8px;align-items:center}

/* Top Visual Panels */
.viz-panel{
  background:#fff;border:1.5px solid #2b3e52;border-radius:10px;
  padding:5px 9px 4px;height:100%;display:flex;flex-direction:column;justify-content:space-between;
}
.panel-title{
  font-size:12.5px;font-weight:850;color:#152b42;letter-spacing:.2px;
  display:flex;align-items:center;justify-content:space-between;gap:6px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:2px;
}
.small-muted{font-size:10px;font-weight:500;color:#5c748b;letter-spacing:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.posture-svg{display:block;width:100%;height:auto;max-height:178px;margin:auto}
.svg-caption{font:700 11.5px Arial,sans-serif;fill:#35516c;letter-spacing:1px}
.svg-legend{font:10px Arial,sans-serif;fill:#657e96}
.floor-line{stroke:#a8bbcb;stroke-width:1.5}
.chair-back{stroke:#7f9bb3;stroke-width:13;stroke-linecap:round}
.chair-leg{stroke:#6b889e;stroke-width:9;stroke-linecap:round}
.person-neck{stroke:#e7ad84;stroke-width:10;stroke-linecap:round}
.person-head{fill:#f0bf98;stroke:#dd9a70;stroke-width:1.5}
.person-arm{stroke:#e3a880;stroke-width:13;stroke-linecap:round;fill:none}
.person-leg{stroke:#283e62;stroke-width:15;stroke-linecap:round;stroke-linejoin:round;fill:none}
.person-pants{stroke:#283e62;stroke-width:22;stroke-linecap:round;fill:none}
.breathing{transform-origin:center;animation:softBreathe 3s ease-in-out infinite alternate}
.piston-pulse{animation:pistonGlow 1.1s ease-in-out infinite alternate}
@keyframes softBreathe{0%{transform:translateY(0px)}100%{transform:translateY(-2px)}}
@keyframes pistonGlow{0%{opacity:.55}100%{opacity:1}}

/* ACTUATOR Outer Box & Point Sub-Boxes */
.section-frame{
  background:#ffffff;border:1.5px solid #2b3e52;border-radius:10px;padding:5px 10px 7px;margin-top:1px;
}
.section-heading{
  text-align:center;font-size:14px;font-weight:850;color:#12283e;letter-spacing:.8px;margin-bottom:4px;
}
.point-grid-3{
  display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;
}
.point-box{
  border:1.5px solid #3c5269;border-radius:8px;padding:6px 9px;background:#ffffff;
  display:flex;flex-direction:column;justify-content:space-between;min-width:0;
}
.point-title-big{
  font-size:15.5px;font-weight:850;color:#0f263c;text-align:center;
  margin-bottom:5px;line-height:1.2;letter-spacing:-.1px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.actuator-inner{
  background:#f8fbfd;border:1px solid #d8e4ef;border-radius:8px;padding:6px 8px;
}
.contact-top{
  display:flex;justify-content:space-between;align-items:center;gap:6px;margin-bottom:4px;flex-wrap:nowrap;
}
.contact-zone{
  font-size:11px;font-weight:800;color:#223c56;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.act-state{font-size:9.5px;font-weight:800;padding:2px 7px;border-radius:999px;white-space:nowrap;flex-shrink:0}
.act-ext{background:#e0f2fe;color:#0369a1;border:1px solid #bae6fd}
.act-ret{background:#fef3c7;color:#b45309;border:1px solid #fde68a}
.act-hold{background:#e6f7ef;color:#147a52;border:1px solid #b4e6ce}
.act-stop{background:#f1f5f9;color:#64748b;border:1px solid #cbd5e1}

.kpi-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px}
.kpi{background:#ffffff;border:1px solid #e3ecf4;border-radius:6px;padding:4px 7px;min-width:0}
.kpi-label{font-size:10.5px;font-weight:700;color:#546c83;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi-value{font-size:22px;font-weight:900;color:#0f263c;white-space:nowrap;line-height:1.15;letter-spacing:-.4px}
.kpi-value small{font-size:12px;font-weight:750;color:#4f667d;margin-left:1px}

/* Progress bars */
.metric{margin-top:5px}
.metric-head{display:flex;justify-content:space-between;align-items:baseline;gap:6px;
  font-size:10.5px;color:#4a6279;font-weight:750;white-space:nowrap}
.metric-head span:first-child{overflow:hidden;text-overflow:ellipsis}
.metric-val{font-size:17px;font-weight:900;color:#0f263c;white-space:nowrap;letter-spacing:-.3px}
.metric-val small{font-size:10.5px;color:#5c748b;font-weight:700;margin-left:2px}
.pbar{position:relative;width:100%;height:9px;background:#e8eef4;border-radius:6px;margin-top:3px}
.pbar-fill{position:absolute;top:0;left:0;height:100%;border-radius:6px;transition:width .6s ease,left .6s ease}
.pbar-marker{position:absolute;top:-3px;width:3px;height:15px;background:#10253a;border-radius:2px;transform:translateX(-1px)}
.pbar-mid{position:absolute;top:-2px;left:50%;width:2px;height:13px;background:#8ea3b6;transform:translateX(-1px)}
.pbar-scale{display:flex;justify-content:space-between;font-size:9px;color:#7a8ea1;margin-top:1px}
.comfort-dots{display:inline-flex;gap:3px;vertical-align:middle;margin-left:4px}
.comfort-dots i{width:9px;height:9px;border-radius:50%;background:#dde6ee;display:inline-block}
.comfort-dots i.on{background:#22aaa6}
.contact-sub{
  font-size:10px;color:#587087;margin-top:4px;line-height:1.25;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}

/* Bottom Row: Agent 1 AI Reasoning & Agent 2 AI Reasoning */
.reasoning-grid-2{
  display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:1px;
}
.reasoning-banner{
  background:#f3f8fc;border:1px solid #d5e3ef;border-radius:7px;padding:4px 8px;margin-bottom:6px;
  font-size:11px;line-height:1.3;color:#233d56;
}
.reasoning-banner-top{
  display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:2px;
}
.agent-badge{
  font-size:9.5px;color:#197a67;background:#e4f6f2;border:1px solid #b6e8da;border-radius:999px;
  padding:1px 7px;white-space:nowrap;font-weight:800;
}
.reasoning-text{
  font-size:10.8px;color:#344e68;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
}
.bullet-list{
  margin:2px 0 0;padding-left:15px;font-size:11px;color:#1e364f;line-height:1.38;
}
.bullet-list li{margin-bottom:2px}
.force-val-highlight{font-size:13.5px;font-weight:850;color:#102a43}
.target-muted{font-size:11px;font-weight:650;color:#526b82}
.bar-bg{width:100%;height:5px;background:#edf2f7;border-radius:4px;margin-top:3px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px}
.contact-badge{font-size:9.5px;font-weight:800;padding:1px 6px;border-radius:999px;display:inline-block;margin-top:2px}
.badge-good{background:#e6f7ef;color:#147a52;border:1px solid #b4e6ce}
.badge-fair{background:#fff6e0;color:#9c6514;border:1px solid #f2d493}
.badge-poor{background:#fdecec;color:#b42b2b;border:1px solid #f3bcbc}
.zone-ai-note{
  font-size:9.8px;color:#48627a;margin-top:3px;font-style:italic;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}

@media(max-width:1150px){
  .point-grid-3,.reasoning-grid-2{grid-template-columns:1fr}
}
@media(prefers-reduced-motion:reduce){.breathing,.piston-pulse{animation:none}}
</style>""", unsafe_allow_html=True)


def init():
    if "chair" not in st.session_state:
        st.session_state.chair = new_state()
    if "runtime" not in st.session_state:
        st.session_state.runtime = {
            "running": True, "last_wall": time.monotonic(),
            "executor": ThreadPoolExecutor(max_workers=2, thread_name_prefix="ergoai_ollama"),
            "occupant_future": None, "chair_future": None,
            "next_shift": SHIFT_SECONDS, "need_chair": True,
            "retry_chair": 0, "revision": 0,
            "a1_error": "", "a2_error": "",
            "a1_state": "Ready · Seated", "a2_state": "Ready · Supervising",
            "good_ticks": 0,
            "prev_positions": [0.0] * 6,
        }
    # Baselines stored in kPa (8.75 kPa = 35 PU, 7.50 kPa = 30 PU, 6.25 kPa = 25 PU)
    for key, default in (("pelvic_kpa", 8.75), ("lumbar_kpa", 7.50), ("thoracic_kpa", 6.25)):
        if key not in st.session_state:
            st.session_state[key] = default
    if "target_contact_pct" not in st.session_state:
        st.session_state.target_contact_pct = 45
    if "seat_occ_toggle" not in st.session_state:
        st.session_state.seat_occ_toggle = True
    if "fault_toggle" not in st.session_state:
        st.session_state.fault_toggle = False
    if "load_slider" not in st.session_state:
        st.session_state.load_slider = 1.0


def fmt_kpa(pu_val: float, signed: bool = False) -> str:
    converted = pu_val * KPA_FACTOR
    fmt = f"{converted:+.{KPA_DECIMALS}f}" if signed else f"{converted:.{KPA_DECIMALS}f}"
    return f"{fmt} kPa"


def snapshot():
    return copy.deepcopy(st.session_state.chair)


def sync_hardware_inputs():
    s = st.session_state.chair
    s["seat_present"] = bool(st.session_state.seat_occ_toggle)
    s["fault"] = bool(st.session_state.fault_toggle)
    s["load"] = float(st.session_state.load_slider)
    if not s["fault"] and s["emergency"] and s["status"].startswith("SENSOR FAULT"):
        reset_emergency(s)


def resync_sliders():
    state = st.session_state.chair
    run = st.session_state.runtime
    pelvic_pu = round(float(st.session_state.pelvic_kpa) / KPA_FACTOR, 1)
    lumbar_pu = round(float(st.session_state.lumbar_kpa) / KPA_FACTOR, 1)
    thoracic_pu = round(float(st.session_state.thoracic_kpa) / KPA_FACTOR, 1)
    desired = [pelvic_pu, pelvic_pu, lumbar_pu, lumbar_pu, thoracic_pu, thoracic_pu]
    if desired != state["base_targets"]:
        state["base_targets"] = desired
        state["targets"] = desired.copy()
        run["need_chair"] = True
        run["revision"] += 1
        run["retry_chair"] = 0
        run["a2_state"] = "Replanning slider targets"


def reset_everything():
    run = st.session_state.runtime
    run["revision"] += 1
    run.update({"running": True, "occupant_future": None, "chair_future": None,
                "last_wall": time.monotonic(), "next_shift": SHIFT_SECONDS,
                "need_chair": True, "retry_chair": 0, "good_ticks": 0,
                "prev_positions": [0.0] * 6,
                "a1_error": "", "a2_error": "",
                "a1_state": "Ready · Seated", "a2_state": "Ready · Supervising"})
    st.session_state.seat_occ_toggle = True
    st.session_state.fault_toggle = False
    st.session_state.load_slider = 1.0
    st.session_state.target_contact_pct = 45
    st.session_state.chair = new_state()
    resync_sliders()


def toggle_run():
    run = st.session_state.runtime
    if not st.session_state.chair["emergency"]:
        run["running"] = not run["running"]
        run["last_wall"] = time.monotonic()


def stop_now():
    emergency_stop(st.session_state.chair)
    st.session_state.runtime["running"] = False
    st.session_state.runtime["revision"] += 1
    st.session_state.runtime["occupant_future"] = None
    st.session_state.runtime["chair_future"] = None


def shorten(value: object, n: int = 220) -> str:
    s = str(value or "")
    return escape(s[:n], quote=True)


def pbar(value: float, vmax: float, color: str, *, vmin: float = 0.0,
         marker: float | None = None, scale: tuple[str, str] | None = None) -> str:
    """HTML progress bar with optional target marker and min/max scale labels."""
    span = max(1e-9, vmax - vmin)
    pct = max(0.0, min(100.0, (value - vmin) / span * 100.0))
    mk = ""
    if marker is not None:
        mpct = max(0.0, min(100.0, (marker - vmin) / span * 100.0))
        mk = f'<div class="pbar-marker" style="left:{mpct:.1f}%" title="Target"></div>'
    sc = f'<div class="pbar-scale"><span>{scale[0]}</span><span>{scale[1]}</span></div>' if scale else ""
    return (f'<div class="pbar"><div class="pbar-fill" style="width:{pct:.1f}%;background:{color}"></div>{mk}</div>{sc}')


def lateral_bar(bias_pu: float, single_pad: bool) -> str:
    """Center-zero bar: fill grows left or right from the middle."""
    if single_pad:
        return ('<div class="pbar"><div class="pbar-fill" style="left:0;width:100%;background:#cfdbe6"></div></div>'
                '<div class="pbar-scale"><span>Single central pad</span><span>no L/R split</span></div>')
    max_b = 4.0  # kPa full-scale
    b_kpa = max(-max_b, min(max_b, bias_pu * KPA_FACTOR))
    w = abs(b_kpa) / max_b * 50.0
    left = 50.0 if b_kpa >= 0 else 50.0 - w
    col = "#22aaa6" if abs(b_kpa) < 0.5 else "#e59a28" if abs(b_kpa) < 1.5 else "#d84343"
    return (f'<div class="pbar"><div class="pbar-fill" style="left:{left:.1f}%;width:{max(w, 1.0):.1f}%;background:{col}"></div>'
            f'<div class="pbar-mid"></div></div><div class="pbar-scale"><span>Left</span><span>Right</span></div>')


def calc_zone_contact_pct(s: dict, idx1: int, idx2: int) -> float:
    if not occupied(s):
        return 0.0
    mean_pu = (s["pressures"][idx1] + s["pressures"][idx2]) / 2.0
    mean_pos = (s["positions"][idx1] + s["positions"][idx2]) / 2.0
    contact = (mean_pu * 1.15) + (mean_pos * 0.25)
    return round(max(0.0, min(95.0, contact)), 1)


def classify_contact(pct: float, target_pct: float) -> tuple[str, str, str]:
    if pct >= target_pct:
        return f"OPTIMAL (≥{target_pct:.0f}%)", "badge-good", "#169c6c"
    if pct >= max(15.0, target_pct - 15.0):
        return f"NEAR TARGET (<{target_pct:.0f}%)", "badge-fair", "#e59a28"
    return f"LOW CONTACT (<{max(15.0, target_pct - 15.0):.0f}%)", "badge-poor", "#d84343"


def get_three_point_sensors(s: dict) -> list[dict]:
    """Aggregate internal 6-zone state into the 3-Point Simulation Sensor Array."""
    specs = [
        ("Point 1: Lower (Pelvic)", "ACTUATOR 1 - LOWER (PELVIC)", 0, 1),
        ("Point 2: Middle (Lumbar)", "ACTUATOR 2 - LUMBAR (SPINAL)", 2, 3),
        ("Point 3: Upper (Thoracic)", "ACTUATOR 3 - UPPER (THORACIC)", 4, 5),
    ]
    points = []
    for point_title, act_title, i1, i2 in specs:
        p_left, p_right = s["pressures"][i1], s["pressures"][i2]
        t_left, t_right = s["targets"][i1], s["targets"][i2]
        b_left, b_right = s["base_targets"][i1], s["base_targets"][i2]
        pos_l, pos_r = s["positions"][i1], s["positions"][i2]
        mean_pu = (p_left + p_right) / 2.0
        mean_target_pu = (t_left + t_right) / 2.0
        mean_base_pu = (b_left + b_right) / 2.0
        mean_pos_pct = (pos_l + pos_r) / 2.0
        fsr_volts = round(min(3.30, max(0.0, (mean_pu / 70.0) * 3.30)), 2) if occupied(s) else 0.0
        lr_bias = 0.0 if i1 == 4 else round(p_right - p_left, 1)
        points.append({
            "point_title": point_title,
            "act_title": act_title,
            "idx_pair": (i1, i2),
            "single_pad": i1 == 4,
            "mean_pu": mean_pu,
            "target_pu": mean_target_pu,
            "base_pu": mean_base_pu,
            "pos_pct": mean_pos_pct,
            "fsr_v": fsr_volts,
            "lr_bias": lr_bias,
            "contact_pct": calc_zone_contact_pct(s, i1, i2),
        })
    return points


def actuator_side_animation(s: dict, r: dict) -> str:
    """Render Agent 2 visual: Side-view of the seated character with the 3 mechanical actuators
    extending/retracting in real time against the back, with spacious layout so text never overlaps.
    """
    points = get_three_point_sensors(s)
    prev_pos = r.get("prev_positions", [0.0] * 6)
    is_occ = occupied(s)
    posture = str(s.get("posture", "Upright"))

    # Shift coordinates right (+66px) so left-side actuator labels have 160px of clear room
    base_sx, base_sy, base_hx, base_hy = SIDE.get(posture, SIDE["Upright"])
    dx = 66.0

    act_order = [
        ("Pt 3 · Thoracic", points[2], 125.0),
        ("Pt 2 · Lumbar", points[1], 158.0),
        ("Pt 1 · Pelvic", points[0], 191.0),
    ]

    act_elements = []
    pushes = []
    for short_lbl, pt, y_pos in act_order:
        i1, i2 = pt["idx_pair"]
        prev_mean = (prev_pos[i1] + prev_pos[i2]) / 2.0
        curr_mean = pt["pos_pct"] if is_occ else 0.0
        step_rate = round(curr_mean - prev_mean, 2)
        stroke_mm = round((curr_mean / MAX_POSITION_PCT) * MAX_STROKE_MM, 1)

        travel_px = (curr_mean / MAX_POSITION_PCT) * 24.0
        pushes.append(travel_px)
        rod_x2 = 210.0 + travel_px

        if s["emergency"] or not is_occ:
            col, fill_col, state_txt = "#64748b", "#cbd5e1", "■ STOP"
        elif step_rate > 0.15:
            col, fill_col, state_txt = "#0284c7", "#7dd3fc", f"▲ +{step_rate:.1f}%/s"
        elif step_rate < -0.15:
            col, fill_col, state_txt = "#d97706", "#fde047", f"▼ {step_rate:.1f}%/s"
        else:
            col, fill_col, state_txt = "#16a34a", "#86efac", "● HOLD"

        act_elements.append(
            f'<g>'
            f'<text x="8" y="{y_pos - 2:.1f}" font-family="Arial" font-size="10" font-weight="800" fill="#152b42">{short_lbl}</text>'
            f'<text x="8" y="{y_pos + 10:.1f}" font-family="Arial" font-size="9.5" font-weight="700" fill="{col}">{stroke_mm:.1f}mm ({curr_mean:.0f}%) {state_txt}</text>'
            f'<rect x="166" y="{y_pos - 9:.1f}" width="40" height="18" rx="4" fill="#334155" stroke="#1e293b" stroke-width="1.2"/>'
            f'<rect x="169" y="{y_pos - 6:.1f}" width="{max(3.0, (curr_mean / MAX_POSITION_PCT) * 34.0):.1f}" height="12" rx="2" fill="{fill_col}" class="piston-pulse"/>'
            f'<line x1="206" y1="{y_pos:.1f}" x2="{rod_x2:.1f}" y2="{y_pos:.1f}" stroke="{col}" stroke-width="6" stroke-linecap="round"/>'
            f'<rect x="{rod_x2 - 2:.1f}" y="{y_pos - 9:.1f}" width="8" height="18" rx="4" fill="#eeb867" stroke="{col}" stroke-width="1.5"/>'
            f'<circle cx="{rod_x2 + 8:.1f}" cy="{y_pos:.1f}" r="3" fill="{col}" class="piston-pulse"/>'
            f'</g>'
        )

    u_push, m_push, l_push = pushes[0] * 0.25, pushes[1] * 0.30, pushes[2] * 0.20
    hip_x = 163.0 + dx + l_push
    shoulder_x = base_sx + dx + u_push
    shoulder_y = float(base_sy)
    ctrl_x = (shoulder_x - 8.0) + m_push
    ctrl_y = shoulder_y + 48.0
    head_x = base_hx + dx + u_push
    head_y = float(base_hy)
    person_opacity = "1.0" if is_occ else "0.30"
    label = escape(posture)

    raw_html = f"""<div class="viz-panel">
<div class="panel-title">Agent 2 · ACTUATORS IN ACTION · SIDE VIEW <span class="small-muted">0–50 mm Stroke · {label}</span></div>
<svg class="posture-svg" viewBox="0 0 390 280" role="img" aria-label="Side view of actuators in action with seated character">
<defs>
<linearGradient id="act-shirt" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="#56bac0"/><stop offset="100%" stop-color="#197e92"/>
</linearGradient>
<linearGradient id="act-seat" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="#46627a"/><stop offset="100%" stop-color="#263c54"/>
</linearGradient>
</defs>
<text x="195" y="20" text-anchor="middle" class="svg-caption" fill="#35516c" font-size="11" font-family="Arial" font-weight="700">3-POINT PISTONS + OCCUPANT ({label.upper()})</text>
<line x1="12" y1="258" x2="378" y2="258" class="floor-line" stroke="#a8bbcb" stroke-width="1.5"/>
<path d="M212 67 L212 207" class="chair-back" stroke="#8aa2b5" stroke-width="13" stroke-linecap="round"/>
<rect x="211" y="204" width="119" height="14" rx="7" fill="url(#act-seat)"/>
<line x1="250" y1="216" x2="250" y2="255" class="chair-leg" stroke="#597087" stroke-width="9" stroke-linecap="round"/>
<line x1="231" y1="255" x2="272" y2="255" class="chair-leg" stroke="#597087" stroke-width="9" stroke-linecap="round"/>
{"".join(act_elements)}
<g class="breathing" opacity="{person_opacity}">
<path d="M{hip_x:.1f} 194 Q{ctrl_x:.1f} {ctrl_y:.1f} {shoulder_x:.1f} {shoulder_y:.1f}" stroke="url(#act-shirt)" stroke-width="30" fill="none" stroke-linecap="round"/>
<path d="M{hip_x - 4:.1f} 192 Q{ctrl_x - 5:.1f} {ctrl_y:.1f} {shoulder_x - 4:.1f} {shoulder_y + 4:.1f}" stroke="#ffffff" stroke-width="2.5" stroke-dasharray="4,3" fill="none"/>
<path d="M{shoulder_x - 3:.1f} {shoulder_y + 19:.1f} Q{shoulder_x + 12:.1f} {shoulder_y + 44:.1f} {shoulder_x + 22:.1f} {shoulder_y + 59:.1f}" class="person-arm" stroke="#e3a880" stroke-width="13" stroke-linecap="round" fill="none"/>
<path d="M{hip_x - 1:.1f} 194 L312 194 L308 238 L308 249" class="person-leg" stroke="#283e62" stroke-width="15" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
<path d="M{hip_x:.1f} 196 L301 199" class="person-pants" stroke="#283e62" stroke-width="22" stroke-linecap="round" fill="none"/>
<line x1="{shoulder_x:.1f}" y1="{shoulder_y:.1f}" x2="{head_x:.1f}" y2="{head_y + 18:.1f}" class="person-neck" stroke="#e7ad84" stroke-width="10" stroke-linecap="round"/>
<circle cx="{head_x:.1f}" cy="{head_y:.1f}" r="21" class="person-head" fill="#f0bf98" stroke="#dd9a70" stroke-width="1.5"/>
<circle cx="{head_x + 9:.1f}" cy="{head_y - 2:.1f}" r="2" fill="#294353"/>
</g>
<text x="195" y="273" text-anchor="middle" class="svg-legend" fill="#657e96" font-size="10" font-family="Arial">Pistons push pads into Pelvic, Lumbar &amp; Thoracic spine</text>
</svg>
</div>"""
    return "\n".join(line.strip() for line in raw_html.splitlines() if line.strip())


def process_futures():
    s, r = st.session_state.chair, st.session_state.runtime
    for role in ("occupant", "chair"):
        name = f"{role}_future"
        pending = r[name]
        if pending is None:
            continue
        future, revision = pending
        if not future.done():
            continue
        r[name] = None
        if not r["running"] or s["emergency"]:
            continue
        try:
            output = future.result()
            if revision != r["revision"]:
                if role == "chair":
                    r["need_chair"] = True
                continue
            if role == "occupant":
                prov = getattr(output, "provider", None) or f"Ollama / {MODEL}"
                applied = apply_occupant_proposal(
                    s, output.next_posture, provider=prov,
                    reason=output.reason, comfort_score=output.comfort_score,
                )
                r["a1_state"] = f"Changed to {applied['New posture']}"
                r["a1_error"] = ""
                r["revision"] += 1
                r["need_chair"] = True
                r["retry_chair"] = 0
            else:
                prov = getattr(output, "provider", None) or f"Ollama / {MODEL}"
                outcome = apply_chair_proposal(
                    s, output.targets, provider=prov,
                    rationale=output.rationale, assessment=output.assessment,
                    zone_reasons=output.zone_reasons,
                )
                r["a2_state"] = "Guardrail-limited" if outcome["Guardrail result"].startswith("LIMITED") else "Plan applied"
                r["a2_error"] = ""
        except Exception as exc:
            msg = str(exc)[:330]
            if role == "occupant":
                r["a1_error"] = msg
                r["a1_state"] = "LLM retry scheduled"
            else:
                r["a2_error"] = msg
                r["a2_state"] = "Using baseline sliders"
                r["need_chair"] = True
                r["retry_chair"] = s["tick"] + 10


def advance_and_schedule():
    s, r = st.session_state.chair, st.session_state.runtime
    sync_hardware_inputs()
    if not r["running"] or s["emergency"]:
        r["last_wall"] = time.monotonic()
        return
    now = time.monotonic()
    elapsed = int(now - r["last_wall"])
    target_c = float(st.session_state.get("target_contact_pct", 45))
    if elapsed:
        for _ in range(min(4, elapsed)):
            r["prev_positions"] = s["positions"].copy()
            step(s)
            c_low = calc_zone_contact_pct(s, 0, 1)
            c_lum = calc_zone_contact_pct(s, 2, 3)
            c_upp = calc_zone_contact_pct(s, 4, 5)
            if c_low >= target_c and c_lum >= target_c and c_upp >= target_c:
                r["good_ticks"] += 1
        r["last_wall"] = now if elapsed > 4 else r["last_wall"] + elapsed
    if s["emergency"]:
        r["running"] = False
        return
    process_futures()
    if r["chair_future"] is not None or r["occupant_future"] is not None:
        return
    if not occupied(s):
        return
    if r["need_chair"] and s["tick"] >= r["retry_chair"]:
        r["need_chair"] = False
        r["a2_state"] = "AI generating plan..."
        r["chair_future"] = (r["executor"].submit(get_chair_plan, snapshot()), r["revision"])
    elif s["tick"] >= r["next_shift"]:
        r["next_shift"] = s["tick"] + SHIFT_SECONDS
        r["a1_state"] = "AI choosing posture..."
        r["occupant_future"] = (r["executor"].submit(get_occupant_plan, snapshot()), r["revision"])


def pressure_chart(s: dict) -> go.Figure:
    pts = get_three_point_sensors(s)
    labels = ["Pt 1: Pelvic", "Pt 2: Lumbar", "Pt 3: Thoracic"]
    measured = [round(p["mean_pu"] * KPA_FACTOR, 2) for p in pts]
    targets = [round(p["target_pu"] * KPA_FACTOR, 2) for p in pts]
    sliders = [round(p["base_pu"] * KPA_FACTOR, 2) for p in pts]

    y_max = round(70.0 * KPA_FACTOR, 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=measured, name="Live (kPa)", marker_color="#22aaa6"))
    fig.add_trace(go.Scatter(x=labels, y=targets, name="Target (kPa)", mode="lines+markers",
                             marker={"size": 6}, line={"color": "#eb9846", "width": 2.2}))
    fig.add_trace(go.Scatter(x=labels, y=sliders, name="Baseline (kPa)",
                             line={"color": "#8186b8", "width": 1.6, "dash": "dot"}, mode="lines"))
    fig.update_layout(
        title={"text": "<b>3-POINT FORCE (kPa)</b> · <span style='font-size:9.5px'>"
                       "<span style='color:#159c9c'>■</span> Live  "
                       "<span style='color:#e38a33'>●</span> Target  "
                       "<span style='color:#8186b8'>···</span> Base</span>",
               "x": 0.03, "y": 0.92, "font": {"size": 11, "color": "#152b42"}},
        height=104, margin={"l": 30, "r": 8, "t": 26, "b": 20},
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font={"color": "#546b82", "size": 9.5},
        yaxis={"range": [0, y_max], "gridcolor": "#eaf0f5", "tickfont": {"size": 8.5}, "fixedrange": True, "zeroline": False},
        xaxis={"tickfont": {"size": 9.5}, "fixedrange": True, "showgrid": False},
        showlegend=False, bargap=.28)
    return fig


def trend_chart(s: dict) -> go.Figure:
    rows = s["history"][-80:]
    x = [e["t (s)"] for e in rows] or [0]
    actual = [round(e["Mean actual (PU)"] * KPA_FACTOR, 2) for e in rows] or [round((sum(s["pressures"]) / 6) * KPA_FACTOR, 2)]
    target = [round(e["Mean target (PU)"] * KPA_FACTOR, 2) for e in rows] or [round((sum(s["targets"]) / 6) * KPA_FACTOR, 2)]
    ext = [e["Mean extension (%)"] for e in rows] or [sum(s["positions"]) / 6]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=actual, mode="lines", name="Live (kPa)", line={"color": "#22aaa6", "width": 2.0}))
    fig.add_trace(go.Scatter(x=x, y=target, mode="lines", name="Target (kPa)", line={"color": "#eb9846", "width": 1.8, "dash": "dot"}))
    fig.add_trace(go.Scatter(x=x, y=ext, mode="lines", name="Actuator (%)", line={"color": "#1d4ed8", "width": 1.4}))
    fig.update_layout(
        title={"text": "<b>TREND (kPa &amp; ACTUATOR %)</b> · <span style='font-size:9.5px'>"
                       "<span style='color:#159c9c'>━</span> kPa  "
                       "<span style='color:#e38a33'>┄</span> Target  "
                       "<span style='color:#1d4ed8'>━</span> Ext %</span>",
               "x": 0.03, "y": 0.92, "font": {"size": 11, "color": "#152b42"}},
        height=104, margin={"l": 30, "r": 8, "t": 26, "b": 20},
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font={"color": "#546b82", "size": 9.5},
        xaxis={"tickfont": {"size": 8.5}, "gridcolor": "#eff3f7", "fixedrange": True, "zeroline": False},
        yaxis={"tickfont": {"size": 8.5}, "gridcolor": "#eaf0f5", "fixedrange": True, "zeroline": False},
        showlegend=False)
    return fig


def render_actuator_section(s: dict, r: dict) -> None:
    """Render the middle ACTUATOR box with bigger Point 1 / Point 2 / Point 3 titles."""
    points = get_three_point_sensors(s)
    prev_pos = r.get("prev_positions", [0.0] * 6)
    boxes_html = []
    for pt in points:
        i1, i2 = pt["idx_pair"]
        prev_mean = (prev_pos[i1] + prev_pos[i2]) / 2.0
        curr_mean = pt["pos_pct"]
        step_rate = round(curr_mean - prev_mean, 2)
        stroke_mm = round((curr_mean / MAX_POSITION_PCT) * MAX_STROKE_MM, 1)
        cmd_rate = max(-MAX_STEP_PCT, min(MAX_STEP_PCT, PROPORTIONAL_GAIN * (pt["target_pu"] - pt["mean_pu"])))
        pwm_duty = round((abs(cmd_rate) / MAX_STEP_PCT) * 100.0, 0) if (occupied(s) and not s["emergency"]) else 0.0

        if s["emergency"] or not occupied(s):
            state_lbl, state_cls = "■ INTERLOCK", "act-stop"
        elif step_rate > 0.15:
            state_lbl, state_cls = f"▲ INFLATING / EXTENDING ({step_rate:+.1f}%/s)", "act-ext"
        elif step_rate < -0.15:
            state_lbl, state_cls = f"▼ DEFLATING / RETRACTING ({step_rate:+.1f}%/s)", "act-ret"
        else:
            state_lbl, state_cls = "● HOLDING TARGET", "act-hold"

        split_txt = ("Single central pad (no L/R)" if pt["single_pad"]
                     else f'L/R Split: {s["positions"][i1]:.1f}% / {s["positions"][i2]:.1f}%')
        bar_col = {"act-ext": "#0284c7", "act-ret": "#d97706", "act-hold": "#16a34a"}.get(state_cls, "#94a3b8")
        boxes_html.append(f'''<div class="point-box">
          <div class="point-title-big">{pt["point_title"]}</div>
          <div class="actuator-inner">
            <div class="contact-top">
              <span class="contact-zone">{pt["act_title"]}</span>
              <span class="act-state {state_cls}">{state_lbl}</span>
            </div>
            <div class="kpi-row">
              <div class="kpi"><div class="kpi-label">Extension</div><div class="kpi-value">{curr_mean:.1f}<small>%</small></div></div>
              <div class="kpi"><div class="kpi-label">Linear Stroke</div><div class="kpi-value">{stroke_mm:.1f}<small>mm</small></div></div>
              <div class="kpi"><div class="kpi-label">PWM Drive</div><div class="kpi-value">{pwm_duty:.0f}<small>%</small></div></div>
            </div>
            {pbar(curr_mean, MAX_POSITION_PCT, bar_col, scale=("0 mm", f"{MAX_STROKE_MM:.0f} mm max ({MAX_POSITION_PCT:.0f}%)"))}
            <div class="contact-sub">{split_txt}</div>
          </div>
        </div>''')

    st.markdown(f'''<div class="section-frame">
      <div class="section-heading">ACTUATOR</div>
      <div class="point-grid-3">{"".join(boxes_html)}</div>
    </div>''', unsafe_allow_html=True)


def render_ai_reasoning_section(s: dict, r: dict) -> None:
    """Render Bottom Row: Agent 1 AI reasoning (left) and Agent 2 AI reasoning (right)
    with Point 1 / 2 / 3 sub-boxes and full AI reasoning showcase.
    """
    points = get_three_point_sensors(s)
    target_contact = float(st.session_state.get("target_contact_pct", 45))

    latest_occ = s["occupant_decisions"][-1] if s["occupant_decisions"] else None
    latest_chair = s["chair_decisions"][-1] if s["chair_decisions"] else None

    occ_reason = latest_occ["Reason"] if latest_occ else "Seated comfortably; Ollama will autonomously shift posture at the next interval."
    comfort = f"{latest_occ['Fictional comfort (1-5)']}/5" if latest_occ else "4/5"
    a1_status = shorten(r["a1_error"] if r["a1_error"] else r["a1_state"], 120)
    a1_badge = "THINKING…" if r["occupant_future"] else f"POSTURE: {s['posture'].upper()} · COMFORT {comfort}"

    assessment = latest_chair["Assessment"] if latest_chair else "Monitoring 3-point kPa pressure distribution across Pelvic, Lumbar, and Thoracic zones."
    why = latest_chair["Justification"] if latest_chair else "Maintaining operator kPa baselines until next AI supervisor evaluation."
    guard = latest_chair["Guardrail result"] if latest_chair else "Active"
    a2_status = shorten(r["a2_error"] if r["a2_error"] else r["a2_state"], 120)
    a2_badge = "THINKING…" if r["chair_future"] else ("GUARDRAIL LIMITED" if str(guard).startswith("LIMITED") else "SYNCHRO ACTIVE")

    zone_reasons = []
    if latest_chair and "zones" in latest_chair and len(latest_chair["zones"]) == 6:
        zone_reasons = [
            latest_chair["zones"][0].get("Explanation", ""),
            latest_chair["zones"][2].get("Explanation", ""),
            latest_chair["zones"][4].get("Explanation", ""),
        ]
    else:
        zone_reasons = [
            "Balanced pelvic foundation.",
            "Supporting lumbar lordosis.",
            "Aligning upper thoracic contact.",
        ]

    # Agent 1 Point Cards (Live exert force by the person)
    max_force_kpa = 70.0 * KPA_FACTOR  # sensor full-scale (17.5 kPa)
    a1_cards = []
    for idx, pt in enumerate(points):
        if pt["single_pad"]:
            bias_txt = "Central"
        elif abs(pt["lr_bias"]) < 1.0:
            bias_txt = "Balanced"
        else:
            bias_txt = f"{'R' if pt['lr_bias'] > 0 else 'L'} +{abs(pt['lr_bias']) * KPA_FACTOR:.2f} kPa"
        force_kpa = pt["mean_pu"] * KPA_FACTOR
        a1_cards.append(f'''<div class="point-box">
          <div class="point-title-big">{pt["point_title"]}</div>
          <div class="metric">
            <div class="metric-head"><span>Live exert force by the person</span></div>
            <div class="metric-val">{force_kpa:.2f}<small>kPa</small></div>
            {pbar(force_kpa, max_force_kpa, "#22aaa6", scale=("0", f"{max_force_kpa:.1f} kPa"))}
          </div>
          <div class="metric">
            <div class="metric-head"><span>Sensor FSR</span><span>{pt["fsr_v"]:.2f} V</span></div>
            {pbar(pt["fsr_v"], 3.3, "#6366f1")}
          </div>
          <div class="metric">
            <div class="metric-head"><span>Lateral load</span><span>{bias_txt}</span></div>
            {lateral_bar(pt["lr_bias"], pt["single_pad"])}
          </div>
        </div>''')

    # Agent 2 Point Cards (Target Force & Live Contact point (Target Contact point))
    t_min_kpa, t_max_kpa = 15.0 * KPA_FACTOR, 55.0 * KPA_FACTOR
    a2_cards = []
    for idx, pt in enumerate(points):
        c_pct = pt["contact_pct"]
        badge_txt, badge_cls, bar_col = classify_contact(c_pct, target_contact)
        z_note = shorten(zone_reasons[idx], 65)
        tgt_kpa = pt["target_pu"] * KPA_FACTOR
        base_kpa = pt["base_pu"] * KPA_FACTOR
        delta = tgt_kpa - base_kpa
        a2_cards.append(f'''<div class="point-box">
          <div class="point-title-big">{pt["point_title"]}</div>
          <div class="metric">
            <div class="metric-head"><span>Target Force</span><span>{delta:+.2f} vs base</span></div>
            <div class="metric-val">{tgt_kpa:.2f}<small>kPa</small></div>
            {pbar(tgt_kpa, t_max_kpa, "#eb9846", vmin=t_min_kpa, marker=base_kpa, scale=(f"{t_min_kpa:.2f}", f"{t_max_kpa:.2f} kPa"))}
          </div>
          <div class="metric">
            <div class="metric-head"><span>Live Contact (Target {target_contact:.0f}%)</span></div>
            <div class="metric-val">{c_pct:.1f}<small>%</small></div>
            {pbar(c_pct, 100.0, bar_col, marker=target_contact)}
          </div>
          <div style="margin-top:4px"><span class="contact-badge {badge_cls}">{badge_txt}</span></div>
          <div class="zone-ai-note" title="{escape(str(zone_reasons[idx]))}">AI: {z_note}</div>
        </div>''')

    comfort_n = int(latest_occ["Fictional comfort (1-5)"]) if latest_occ else 4
    dots = "".join(f'<i class="{"on" if k < comfort_n else ""}"></i>' for k in range(5))
    a1_badge = "THINKING…" if r["occupant_future"] else f"POSTURE: {s['posture'].upper()}"

    prov_a1 = latest_occ.get("Source", "AI Occupant") if latest_occ else "AI Occupant"
    prov_a2 = latest_chair.get("Source", "AI Supervisor") if latest_chair else "AI Supervisor"

    st.markdown(f'''<div class="reasoning-grid-2">
      <div class="section-frame" style="margin-top:0">
        <div class="section-heading">Agent 1 AI reasoning</div>
        <div class="reasoning-banner">
          <div class="reasoning-banner-top">
            <strong>{shorten(prov_a1, 35)} ({shorten(a1_status, 45)})</strong>
            <span class="agent-badge">{a1_badge}</span>
          </div>
          <div class="reasoning-text" title="{shorten(occ_reason, 500)}">💬 &ldquo;{shorten(occ_reason, 185)}&rdquo;</div>
          <div class="metric-head" style="margin-top:3px;justify-content:flex-start">Comfort {comfort}<span class="comfort-dots">{dots}</span></div>
        </div>
        <div class="point-grid-3">{"".join(a1_cards)}</div>
      </div>
      <div class="section-frame" style="margin-top:0">
        <div class="section-heading">Agent 2 AI reasoning</div>
        <div class="reasoning-banner">
          <div class="reasoning-banner-top">
            <strong>{shorten(prov_a2, 35)} ({shorten(a2_status, 45)})</strong>
            <span class="agent-badge">{a2_badge}</span>
          </div>
          <div class="reasoning-text" title="Assessment: {shorten(assessment, 400)} | Why: {shorten(why, 400)}">
            🧠 <strong>Assessment:</strong> {shorten(assessment, 95)} &nbsp;|&nbsp; <strong>Why:</strong> {shorten(why, 100)}
          </div>
        </div>
        <div class="point-grid-3">{"".join(a2_cards)}</div>
      </div>
    </div>''', unsafe_allow_html=True)


init()

# 1. Compact Header Bar
hdr_left, hdr_right = st.columns([6.0, 4.0], gap="small")
with hdr_left:
    s_init = st.session_state.chair
    r_init = st.session_state.runtime
    st.markdown(
        f'''<div class="ergo-header">
          <span class="ergo-logo">ERGO <em>AI</em></span>
          <span class="ergo-sub">Adaptive Synchro System · 3-Point Sensor &amp; Actuator Touchpoint · {shorten(MODEL, 35)}</span>
        </div>''',
        unsafe_allow_html=True,
    )
with hdr_right:
    b1, b2, b3, t1, t2 = st.columns([0.9, 0.85, 0.95, 1.15, 1.05], gap="small")
    with b1:
        st.button("⏸ Pause" if st.session_state.runtime["running"] else "▶ Resume",
                  on_click=toggle_run, disabled=st.session_state.chair["emergency"],
                  use_container_width=True)
    with b2:
        st.button("↻ Reset", on_click=reset_everything, use_container_width=True)
    with b3:
        st.button("■ E-STOP", on_click=stop_now, type="primary", use_container_width=True)
    with t1:
        st.checkbox("Occupied", key="seat_occ_toggle")
    with t2:
        st.checkbox("Fault", key="fault_toggle")

# 2. Five Sliders Row (Pelvic Baseline, Lumbar Baseline, Thoracic Baseline, Occupant, Target Contact Point %)
sl1, sl2, sl3, sl4, sl5 = st.columns(5, gap="small")
with sl1:
    st.slider("Pelvic Baseline (kPa)", 3.75, 13.75, key="pelvic_kpa", step=0.25)
with sl2:
    st.slider("Lumbar Baseline (kPa)", 3.75, 13.75, key="lumbar_kpa", step=0.25)
with sl3:
    st.slider("Thoracic Baseline (kPa)", 3.75, 13.75, key="thoracic_kpa", step=0.25)
with sl4:
    st.slider("Occupant Load", 0.6, 1.4, key="load_slider", step=0.05)
with sl5:
    st.slider("Target Contact Point (%)", 25, 75, key="target_contact_pct", step=1)
resync_sliders()


@st.fragment(run_every="1s")
def live():
    advance_and_schedule()
    s, r = st.session_state.chair, st.session_state.runtime

    # Top Section: Animations & Live Graphs at the top
    col_a1_viz, col_a2_viz, col_graphs = st.columns([1.05, 1.0, 0.95], gap="small")
    with col_a1_viz:
        st.markdown(posture_illustration(s["posture"], s["positions"]), unsafe_allow_html=True)
    with col_a2_viz:
        st.markdown(actuator_side_animation(s, r), unsafe_allow_html=True)
    with col_graphs:
        st.plotly_chart(pressure_chart(s), width="stretch",
                        config={"displayModeBar": False, "staticPlot": True})
        st.plotly_chart(trend_chart(s), width="stretch",
                        config={"displayModeBar": False, "staticPlot": True})

    # Middle Section: ACTUATOR (Point 1, Point 2, Point 3)
    render_actuator_section(s, r)

    # Bottom Section: Agent 1 AI reasoning & Agent 2 AI reasoning
    render_ai_reasoning_section(s, r)


live()
