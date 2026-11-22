"""Mascotte de MaintiDoc : robot à casque de chantier, en SVG animé (CSS).

Deux états :
- "idle"     : clignement des yeux, anneau qui tourne lentement, écouteurs qui pulsent ;
- "thinking" : les yeux balaient la visière, l'engrenage du casque tourne, l'anneau accélère.
Les animations sont coupées si l'utilisateur a demandé moins de mouvement.
"""
import base64
import math

STEEL = "#1F2A33"
VALVE = "#1D5C8C"
RULE = "#C9D1D7"
GLOW = "#8CCBF5"      # bleu vanne éclairci : lumière des yeux sur la visière sombre

def _gear_path(cx=100, cy=60, r_out=14, r_in=10.5, teeth=8) -> str:
    """Contour d'engrenage : dents trapézoïdales régulières."""
    pts, step = [], 2 * math.pi / teeth
    for i in range(teeth):
        a = i * step
        for da, r in ((-.30, r_in), (-.17, r_out), (.17, r_out), (.30, r_in)):
            pts.append((cx + r * math.cos(a + da * step * 1.6), cy + r * math.sin(a + da * step * 1.6)))
    return "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z"


_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" class="{mode}">
<defs>
  <linearGradient id="shell" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#DCE3E8"/>
  </linearGradient>
  <linearGradient id="dome" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#FFFFFF"/><stop offset=".65" stop-color="#EEF1F3"/>
    <stop offset="1" stop-color="#D3DAE0"/>
  </linearGradient>
  <linearGradient id="visor" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#2B3945"/><stop offset="1" stop-color="{steel}"/>
  </linearGradient>
  <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
    <feGaussianBlur stdDeviation="2.2" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>
<style>
  * {{ transform-box: fill-box; transform-origin: center; }}
  .ring {{ transform-box: view-box; transform-origin: 100px 100px;
          animation: spin 28s linear infinite; }}
  .eyes {{ animation: blink 5s ease-in-out infinite; }}
  .ear {{ animation: pulse 2.8s ease-in-out infinite; }}
  .thinking .ring {{ animation-duration: 4s; }}
  .thinking .eyes {{ animation: scan 1.6s ease-in-out infinite; }}
  .thinking .gear {{ animation: spin 3s linear infinite; }}
  .thinking .ear {{ animation-duration: 1s; }}
  .compact .ring {{ display: none; }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  @keyframes blink {{ 0%, 92%, 100% {{ transform: scaleY(1); }} 95% {{ transform: scaleY(.1); }} }}
  @keyframes scan {{ 0%, 100% {{ transform: translateX(-7px); }} 50% {{ transform: translateX(7px); }} }}
  @keyframes pulse {{ 0%, 100% {{ opacity: .45; }} 50% {{ opacity: 1; }} }}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
</style>

<!-- anneau orbital avec repères -->
<g class="ring" fill="{valve}">
  <circle cx="100" cy="100" r="92" fill="none" stroke="{valve}" stroke-opacity=".55"
          stroke-width="3" stroke-linecap="round" stroke-dasharray="120 14 60 14 150 14 90 16"/>
  <circle cx="100" cy="8" r="4.5"/><circle cx="192" cy="100" r="4"/>
  <circle cx="35" cy="165" r="4"/>
</g>

<!-- tête -->
<rect x="44" y="84" width="112" height="86" rx="40" fill="url(#shell)" stroke="{rule}" stroke-width="1.5"/>
<!-- écouteurs -->
<rect x="30" y="102" width="20" height="48" rx="9" fill="{steel}"/>
<rect x="150" y="102" width="20" height="48" rx="9" fill="{steel}"/>
<rect class="ear" x="36" y="110" width="4" height="32" rx="2" fill="{glow}" filter="url(#glow)"/>
<rect class="ear" x="160" y="110" width="4" height="32" rx="2" fill="{glow}" filter="url(#glow)"/>
<!-- visière -->
<rect x="54" y="100" width="92" height="58" rx="27" fill="url(#visor)"/>
<path d="M66 108 q34 -8 68 0" fill="none" stroke="#FFFFFF" stroke-opacity=".12" stroke-width="3"
      stroke-linecap="round"/>
<!-- yeux souriants -->
<g class="eyes" fill="none" stroke="{glow}" stroke-width="6" stroke-linecap="round" filter="url(#glow)">
  <path d="M72 134 q11 -13 22 0"/><path d="M106 134 q11 -13 22 0"/>
</g>

<!-- casque -->
<path d="M40 94 C40 50 66 26 100 26 C134 26 160 50 160 94 Z" fill="url(#dome)"
      stroke="{rule}" stroke-width="1.5"/>
<path d="M88 27 L84 90 L116 90 L112 27 Z" fill="#E4E9ED" stroke="{rule}" stroke-width="1.5"/>
<path d="M62 44 Q56 64 58 92 M138 44 Q144 64 142 92" fill="none" stroke="{rule}" stroke-width="2"/>
<rect x="30" y="88" width="140" height="15" rx="7.5" fill="#F7F9FA" stroke="{rule}" stroke-width="1.5"/>
<!-- engrenage -->
<g class="gear">
  <path d="{gear}" fill="{steel}"/>
  <circle cx="100" cy="60" r="4.5" fill="#E4E9ED"/>
</g>
</svg>"""


def mascot_svg(mode: str = "idle") -> str:
    """mode : "idle", "thinking", éventuellement suivi de " compact" (sans anneau)."""
    vb = "26 22 148 152" if "compact" in mode else "0 0 200 200"   # compact : cadrage serré
    return _SVG.format(mode=mode, vb=vb, steel=STEEL, valve=VALVE, rule=RULE, glow=GLOW,
                       gear=_gear_path())


def mascot_b64(mode: str = "idle") -> str:
    """SVG encodé en base64, pour une image de fond CSS (st.html retire les SVG en ligne)."""
    return base64.b64encode(mascot_svg(mode).encode()).decode()


# Avatar de l'utilisateur : tête de technicien, construite comme celle du robot
# (casque, protections auditives, lunettes de sécurité) mais humaine et sans visage
# détaillé. Casque bleu vanne et anneau acier pour le distinguer de l'assistant.
# Statique : seul l'assistant est animé.
_USER_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
<defs>
  <linearGradient id="skin" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#6E7E8B"/><stop offset="1" stop-color="#56656F"/>
  </linearGradient>
  <linearGradient id="hat" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#2E73A8"/><stop offset="1" stop-color="{valve}"/>
  </linearGradient>
</defs>
<!-- anneau (motif différent de celui de l'assistant) -->
<g fill="{steel}" fill-opacity=".55">
  <circle cx="100" cy="100" r="92" fill="none" stroke="{steel}" stroke-opacity=".35"
          stroke-width="3" stroke-linecap="round" stroke-dasharray="200 14 150 14 160 40"/>
  <circle cx="8" cy="100" r="4"/><circle cx="165" cy="35" r="4"/>
</g>
<!-- cou et tête -->
<rect x="84" y="150" width="32" height="26" rx="10" fill="#4A5966"/>
<ellipse cx="100" cy="118" rx="42" ry="50" fill="url(#skin)"/>
<!-- protections auditives -->
<rect x="44" y="102" width="17" height="38" rx="8" fill="{steel}"/>
<rect x="139" y="102" width="17" height="38" rx="8" fill="{steel}"/>
<!-- lunettes de sécurité -->
<rect x="62" y="110" width="76" height="24" rx="11" fill="{steel}" fill-opacity=".92"/>
<path d="M70 116 q30 -5 60 0" fill="none" stroke="#FFFFFF" stroke-opacity=".18"
      stroke-width="2.5" stroke-linecap="round"/>
<!-- casque -->
<path d="M52 98 C52 58 74 36 100 36 C126 36 148 58 148 98 Z" fill="url(#hat)"/>
<path d="M92 37 L89 94 L111 94 L108 37 Z" fill="#FFFFFF" fill-opacity=".16"/>
<rect x="42" y="92" width="116" height="13" rx="6.5" fill="{valve}" stroke="#174B73"
      stroke-width="1.2"/>
</svg>"""


def user_b64() -> str:
    svg = _USER_SVG.format(steel=STEEL, valve=VALVE)
    return base64.b64encode(svg.encode()).decode()
