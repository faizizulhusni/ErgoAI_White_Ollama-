"""Compact vector illustrations of virtual body posture and six virtual pads."""
from __future__ import annotations

from html import escape

# Side-view body coordinates are illustrative, not a biomechanics model.
SIDE = {
    "Upright": (166, 101, 166, 64),
    "Slouched": (193, 115, 215, 86),
    "Leaning left": (172, 103, 177, 66),
    "Leaning right": (174, 103, 177, 66),
    "Reclined": (129, 112, 111, 80),
}
FRONT = {
    "Upright": (490, 100, 490, 65, 0),
    "Slouched": (496, 111, 503, 87, 0),
    "Leaning left": (460, 111, 453, 78, -18),
    "Leaning right": (520, 111, 527, 78, 18),
    "Reclined": (490, 100, 490, 66, 0),
}


def posture_illustration(posture: str, positions: list[float]) -> str:
    shoulder_x, shoulder_y, head_x, head_y = SIDE.get(posture, SIDE["Upright"])
    fr_x, fr_y, fr_head_x, fr_head_y, lean = FRONT.get(posture, FRONT["Upright"])
    # Pad widths communicate current extension. Orange pad glow is illustrative.
    pads = []
    for y, position in zip((191, 158, 125), (positions[0], positions[2], positions[4])):
        thick = 6 + max(0, min(float(position), 70)) / 12
        pads.append(f'<rect x="{148 - thick:.1f}" y="{y}" width="{thick:.1f}" height="18" rx="5" fill="#eeb867" stroke="#f1cc91" stroke-width="1"/>')
    front_pads = []
    # Front support elements behind the person provide actuator feedback.
    # Pelvic and lumbar rows have left/right pads; thoracic is ONE central pad.
    for ix, pos in enumerate(positions[:4]):
        row = ix // 2
        px = 435 if ix % 2 == 0 else 544
        py = (178, 149)[row]
        opacity = 0.22 + min(0.6, max(0, pos) / 95)
        front_pads.append(f'<rect x="{px}" y="{py}" width="10" height="18" rx="5" fill="#f2ab59" opacity="{opacity:.2f}"/>')
    thoracic_opacity = 0.22 + min(0.6, max(0, positions[4]) / 95)
    front_pads.append(f'<rect x="433" y="112" width="122" height="14" rx="7" fill="#f2ab59" opacity="{thoracic_opacity:.2f}"/>')
    label = escape(posture)
    return f'''<div class="viz-panel">
      <div class="panel-title">Agent 1 · LIVE PERSON &amp; CHAIR <span class="small-muted">Posture: {label} · Actuators in amber</span></div>
      <svg class="posture-svg" viewBox="0 0 660 280" role="img"
           aria-label="Animated side and front views of a seated simulated person in {label} posture with actuator support">
        <defs>
          <linearGradient id="human-shirt" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#56bac0"/><stop offset="100%" stop-color="#197e92"/>
          </linearGradient>
          <linearGradient id="seat-blue" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#46627a"/><stop offset="100%" stop-color="#263c54"/>
          </linearGradient>
        </defs>
        <text x="155" y="20" text-anchor="middle" class="svg-caption" fill="#35516c" font-size="11.5" font-family="Arial" font-weight="700">SIDE VIEW</text>
        <text x="490" y="20" text-anchor="middle" class="svg-caption" fill="#35516c" font-size="11.5" font-family="Arial" font-weight="700">FRONT VIEW</text>
        <line x1="15" y1="258" x2="311" y2="258" class="floor-line" stroke="#a8bbcb" stroke-width="1.5"/>
        <line x1="355" y1="258" x2="646" y2="258" class="floor-line" stroke="#a8bbcb" stroke-width="1.5"/>
        <!-- Side view: chair -->
        <path d="M146 67 L146 207" class="chair-back" stroke="#8aa2b5" stroke-width="13" stroke-linecap="round"/>
        <rect x="145" y="204" width="119" height="14" rx="7" fill="url(#seat-blue)"/>
        <line x1="184" y1="216" x2="184" y2="255" class="chair-leg" stroke="#597087" stroke-width="9" stroke-linecap="round"/>
        <line x1="165" y1="255" x2="206" y2="255" class="chair-leg" stroke="#597087" stroke-width="9" stroke-linecap="round"/>
        {''.join(pads)}
        <!-- Side view occupant animated with subtle breathing -->
        <g class="breathing">
          <path d="M163 194 Q{shoulder_x - 8} {shoulder_y + 48} {shoulder_x} {shoulder_y}" stroke="url(#human-shirt)" stroke-width="30" fill="none" stroke-linecap="round"/>
          <path d="M{shoulder_x - 3} {shoulder_y + 19} Q{shoulder_x + 12} {shoulder_y + 44} {shoulder_x + 22} {shoulder_y + 59}" class="person-arm" stroke="#e3a880" stroke-width="13" stroke-linecap="round" fill="none"/>
          <path d="M162 194 L246 194 L242 238 L242 249" class="person-leg" stroke="#283e62" stroke-width="15" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
          <path d="M163 196 L235 199" class="person-pants" stroke="#283e62" stroke-width="22" stroke-linecap="round" fill="none"/>
          <line x1="{shoulder_x}" y1="{shoulder_y}" x2="{head_x}" y2="{head_y + 18}" class="person-neck" stroke="#e7ad84" stroke-width="10" stroke-linecap="round"/>
          <circle cx="{head_x}" cy="{head_y}" r="21" class="person-head" fill="#f0bf98" stroke="#dd9a70" stroke-width="1.5"/>
          <circle cx="{head_x + 9}" cy="{head_y - 2}" r="2" fill="#294353"/>
        </g>
        <!-- Front view: chair -->
        <rect x="426" y="83" width="128" height="120" rx="24" fill="#e4edf5" stroke="#9aafc2" stroke-width="5"/>
        <rect x="419" y="207" width="144" height="14" rx="7" fill="url(#seat-blue)"/>
        <path d="M490 220 L490 256 M465 256 L515 256" class="chair-leg" stroke="#597087" stroke-width="9" stroke-linecap="round"/>
        {''.join(front_pads)}
        <g class="breathing">
          <path d="M490 198 Q{fr_x} {fr_y + 40} {fr_x} {fr_y}" stroke="url(#human-shirt)" stroke-width="53" fill="none" stroke-linecap="round"/>
          <path d="M{fr_x - 20} {fr_y + 8} Q{fr_x - 38 + lean/3} 150 {fr_x - 30 + lean/3} 186" class="person-arm" stroke="#e3a880" stroke-width="13" stroke-linecap="round" fill="none"/>
          <path d="M{fr_x + 20} {fr_y + 8} Q{fr_x + 39 + lean/3} 150 {fr_x + 31 + lean/3} 186" class="person-arm" stroke="#e3a880" stroke-width="13" stroke-linecap="round" fill="none"/>
          <path d="M470 203 L461 251 M510 203 L519 251" class="person-leg" stroke="#283e62" stroke-width="15" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
          <line x1="{fr_x}" y1="{fr_y}" x2="{fr_head_x}" y2="{fr_head_y + 16}" class="person-neck" stroke="#e7ad84" stroke-width="10" stroke-linecap="round"/>
          <circle cx="{fr_head_x}" cy="{fr_head_y}" r="20" class="person-head" fill="#f0bf98" stroke="#dd9a70" stroke-width="1.5"/>
          <circle cx="{fr_head_x - 7}" cy="{fr_head_y - 2}" r="2" fill="#294353"/>
          <circle cx="{fr_head_x + 7}" cy="{fr_head_y - 2}" r="2" fill="#294353"/>
        </g>
        <text x="156" y="273" text-anchor="middle" class="svg-legend" fill="#657e96" font-size="10" font-family="Arial">Contact pads expand as support changes</text>
        <text x="492" y="273" text-anchor="middle" class="svg-legend" fill="#657e96" font-size="10" font-family="Arial">Front tilt follows Agent 1 posture</text>
      </svg>
    </div>'''