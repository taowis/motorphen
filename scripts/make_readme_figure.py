"""Animated README figure, docs/images/motorphen-hero.svg.

Usage: python scripts/make_readme_figure.py

Four scenes on a 32 s loop: masked prediction of each joint, reconstruction
of a normative twin, specificity in held-out healthy adults (E1) and the
paretic side in stroke survivors (E2, E3). Numbers are copied from
results/e1/README.md, results/e2/README.md and results/e3/*_README.md; the
curves, token scores and positions of flagged trials are drawn for
illustration. Plain SVG with CSS animation, so it plays on GitHub.
"""

import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "motorphen-hero.svg"
T = 32.0  # loop length, s
SCENE = 8.0

BG0, BG1, BG2 = "#0d1524", "#111d31", "#16263d"
LINE = "#26364f"
TEXT = "#edf2f9"
SUB = "#9cc3ff"
MUTED = "#8291a8"
NORM = "#7fb8ff"   # normative / model
FLAG = "#ff8a65"   # flagged / deviation
RULE = "#8fd694"   # clinical rule

css = []
_n = [0]


def pct(t):
    return f"{100 * t / T:.2f}%"


def show(on, off, fade=0.35, name=None):
    """Class that is visible from `on` to `off` (s)."""
    _n[0] += 1
    name = name or f"k{_n[0]}"
    a, b = max(on - fade, 0), min(off + fade, T)
    css.append(f".{name}{{opacity:0;animation:{name} {T}s linear infinite}}"
               f"@keyframes {name}{{0%,{pct(a)}{{opacity:0}}{pct(on)},{pct(off)}{{opacity:1}}"
               f"{pct(b)},100%{{opacity:0}}}}")
    return name


def grow(on, dur, off, origin="0% 50%", axis="X"):
    """Scale from 0 to 1 along one axis starting at `on`, hidden after `off`."""
    _n[0] += 1
    name = f"g{_n[0]}"
    css.append(f".{name}{{animation:{name} {T}s cubic-bezier(.3,.7,.2,1) infinite;"
               f"transform-box:fill-box;transform-origin:{origin}}}"
               f"@keyframes {name}{{0%,{pct(on)}{{transform:scale{axis}(0);opacity:1}}"
               f"{pct(on + dur)},{pct(off)}{{transform:scale{axis}(1);opacity:1}}"
               f"{pct(min(off + .35, T))},100%{{transform:scale{axis}(1);opacity:0}}}}")
    return name


def draw(on, dur, off):
    """Stroke drawn along a path with pathLength=1."""
    _n[0] += 1
    name = f"d{_n[0]}"
    css.append(f".{name}{{stroke-dasharray:1;animation:{name} {T}s ease-in-out infinite}}"
               f"@keyframes {name}{{0%,{pct(on)}{{stroke-dashoffset:1;opacity:1}}"
               f"{pct(on + dur)},{pct(off)}{{stroke-dashoffset:0;opacity:1}}"
               f"{pct(min(off + .35, T))},100%{{stroke-dashoffset:0;opacity:0}}}}")
    return name


def pop(on, off):
    _n[0] += 1
    name = f"p{_n[0]}"
    css.append(f".{name}{{animation:{name} {T}s ease-out infinite;transform-box:fill-box;"
               f"transform-origin:center}}"
               f"@keyframes {name}{{0%,{pct(on)}{{opacity:0;transform:scale(.4)}}"
               f"{pct(on + .3)}{{opacity:1;transform:scale(1.25)}}"
               f"{pct(on + .6)},{pct(off)}{{opacity:1;transform:scale(1)}}"
               f"{pct(min(off + .35, T))},100%{{opacity:0}}}}")
    return name


def text(x, y, s, size=14, fill=TEXT, weight=400, cls="sans", anchor="start", extra=""):
    return (f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}"{extra}>{s}</text>')


def header(n, kicker, title):
    return (text(724, 62, f"{n} · {kicker}", 13, FLAG, 700, extra=' letter-spacing="2"')
            + text(724, 88, title, 21, "#dce7f5", 600))


body = []

# ---------------------------------------------------------------- left side
css.append(".rise{animation:rise 1.2s cubic-bezier(.2,.7,.2,1) both}"
           ".d1{animation-delay:.25s}.d2{animation-delay:.5s}"
           "@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}")
css.append(f".prog{{transform-origin:0 0;animation:prog {T}s linear infinite}}"
           "@keyframes prog{from{transform:scaleX(0)}to{transform:scaleX(1)}}")
css.append(".caret{animation:blink 1s steps(1) infinite}@keyframes blink{50%{opacity:0}}")

body.append(f'<g class="rise">{text(92, 190, "MotorPhen", 104, "#f1f5fb", 800, extra=" letter-spacing=\"-3\"")}</g>')
body.append('<g class="rise d1">'
            + text(94, 236, "A Transformer model that names the joint,", 24, SUB, 600)
            + text(94, 267, "tested where the impaired side is known.", 24, SUB, 600) + "</g>")

commands = ["python scripts/02_train_e1.py --seed 1",
            "python scripts/05_eval_e2.py --tags seed1 seed2",
            "python scripts/03_eval_e1.py --tag seed1",
            "python scripts/09_compare_auroc_e2.py"]
cw = 10.8
term = ['<g class="rise d2">',
        f'<rect x="92" y="300" width="560" height="54" rx="12" fill="#0a121e" stroke="{LINE}"/>',
        text(110, 336, "›", 18, FLAG, cls="mono"), '<g clip-path="url(#term)">']
for i, c in enumerate(commands):
    t0 = i * SCENE
    vis = show(t0 + .2, t0 + SCENE - .3, fade=.2)
    L = cw * len(c)
    _n[0] += 1
    ty = f"ty{_n[0]}"
    nch = len(c)
    css.append(f".{ty}{{animation:{ty} {T}s steps({nch}) infinite}}"
               f"@keyframes {ty}{{0%,{pct(t0 + .5)}{{transform:translateX(0)}}"
               f"{pct(t0 + .5 + 0.045 * nch)},100%{{transform:translateX({L:.1f}px)}}}}")
    term.append(f'<g class="{vis}">'
                + text(132, 336, c, 18, "#e4ecf7", cls="mono",
                       extra=f' textLength="{L:.1f}" lengthAdjust="spacingAndGlyphs"')
                + f'<g class="{ty}"><rect x="131" y="316" width="{L + 24:.1f}" height="28" fill="#0a121e"/>'
                  f'<rect class="caret" x="132" y="319" width="10" height="22" rx="2" fill="{NORM}"/></g></g>')
term.append("</g></g>")
body += term
body.append(text(94, 398, "Transformer masked autoencoder · 10.0M parameters · PyTorch", 16, MUTED))
body.append(text(94, 422, "110 able-bodied adults for training · 17 held out · 50 stroke survivors", 16, MUTED))

# ---------------------------------------------------------------- scene 1
s = show(0.15, SCENE - .3, name="s0")
g = [f'<g class="{s}">', header(1, "MASK A JOINT, PREDICT IT", "Which joint does the model not expect?")]

# walking figure
hx, hy = 788, 238
th, sh, ua, fa = 56, 54, 38, 34
css.append(".thigh{animation:thigh 1.2s linear infinite;transform-box:view-box}"
           "@keyframes thigh{0%{transform:rotate(-24deg)}60%{transform:rotate(20deg)}100%{transform:rotate(-24deg)}}"
           ".shank{animation:shank 1.2s linear infinite;transform-box:view-box}"
           "@keyframes shank{0%{transform:rotate(3deg)}15%{transform:rotate(16deg)}40%{transform:rotate(5deg)}"
           "62%{transform:rotate(42deg)}74%{transform:rotate(60deg)}90%{transform:rotate(12deg)}100%{transform:rotate(3deg)}}"
           ".uarm{animation:uarm 1.2s ease-in-out infinite;transform-box:view-box}"
           "@keyframes uarm{0%,100%{transform:rotate(18deg)}50%{transform:rotate(-18deg)}}"
           ".farm{animation:farm 1.2s ease-in-out infinite;transform-box:view-box}"
           "@keyframes farm{0%,100%{transform:rotate(-8deg)}50%{transform:rotate(-30deg)}}"
           ".far{animation-delay:-.6s}"
           ".bob{animation:bob .6s ease-in-out infinite}@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-3px)}}")
flagged_knee = show(5.0, SCENE - .3)


def leg(far):
    col = "#4f6c95" if far else NORM
    d = " far" if far else ""
    kx, ky = hx, hy + th
    knee = (f'<circle cx="{kx}" cy="{ky}" r="5" fill="{col}"/>' if far else
            f'<circle cx="{kx}" cy="{ky}" r="5" fill="{col}"/>'
            f'<circle class="{flagged_knee}" cx="{kx}" cy="{ky}" r="8" fill="{FLAG}"/>')
    return (f'<g class="thigh{d}" style="transform-origin:{hx}px {hy}px">'
            f'<line x1="{hx}" y1="{hy}" x2="{kx}" y2="{ky}" stroke="{col}" stroke-width="7" stroke-linecap="round"/>'
            f'<g class="shank{d}" style="transform-origin:{kx}px {ky}px">'
            f'<line x1="{kx}" y1="{ky}" x2="{kx}" y2="{ky + sh}" stroke="{col}" stroke-width="6" stroke-linecap="round"/>'
            f'<line x1="{kx}" y1="{ky + sh}" x2="{kx + 14}" y2="{ky + sh}" stroke="{col}" stroke-width="5" stroke-linecap="round"/>'
            f'</g>{knee}</g>')


def arm(far):
    col = "#4f6c95" if far else NORM
    d = " far" if far else ""
    sx, sy = hx, hy - 62
    ex, ey = sx, sy + ua
    return (f'<g class="uarm{d}" style="transform-origin:{sx}px {sy}px">'
            f'<line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" stroke="{col}" stroke-width="5" stroke-linecap="round"/>'
            f'<g class="farm{d}" style="transform-origin:{ex}px {ey}px">'
            f'<line x1="{ex}" y1="{ey}" x2="{ex}" y2="{ey + fa}" stroke="{col}" stroke-width="4.5" stroke-linecap="round"/></g></g>')


g.append(f'<line x1="736" y1="{hy + th + sh + 3}" x2="852" y2="{hy + th + sh + 3}" stroke="{LINE}" stroke-width="2"/>')
g.append('<g class="bob">' + leg(True) + arm(True)
         + f'<line x1="{hx}" y1="{hy}" x2="{hx + 4}" y2="{hy - 70}" stroke="{NORM}" stroke-width="7" stroke-linecap="round"/>'
         + f'<circle cx="{hx + 6}" cy="{hy - 88}" r="13" fill="{NORM}"/>'
         + leg(False) + arm(False) + "</g>")
g.append(f'<g class="{flagged_knee}">' + text(806, hy + th + 4, "knee", 13, FLAG, 700) + "</g>")

# token grid: six scored joints x seven frames
joints = ["neck", "pelvis", "L hip", "R hip", "L knee", "R knee"]
score = [0.30, 0.36, 0.46, 0.40, 0.88, 0.52]
gx, gy, c, gap = 948, 128, 17, 4
bx, bw = 1112, 78
thr = 0.62
g.append(text(gx, gy - 12, "7 frames per window", 12, MUTED, cls="mono"))
g.append(text(bx, gy - 12, "score", 12, MUTED, cls="mono"))
for r, (jn, sc) in enumerate(zip(joints, score)):
    y = gy + r * (c + 13)
    hit = r == 4
    g.append(text(gx - 12, y + 13, jn, 13, "#c8d6ea", anchor="end"))
    t_mask = 0.9 + 0.85 * r
    m = show(t_mask, t_mask + 0.75, fade=.15)
    for k in range(7):
        x = gx + k * (c + gap)
        g.append(f'<rect x="{x}" y="{y}" width="{c}" height="{c}" rx="3" fill="{NORM}" fill-opacity="{0.25 + 0.08 * k:.2f}"/>')
    g.append(f'<g class="{m}"><rect x="{gx - 3}" y="{y - 3}" width="{7 * c + 6 * gap + 6}" height="{c + 6}" rx="5" '
             f'fill="{BG1}" stroke="{SUB}" stroke-dasharray="4 3"/>'
             + text(gx + (7 * c + 6 * gap) / 2, y + 13, "withheld", 12, SUB, cls="mono", anchor="middle") + "</g>")
    g.append(f'<rect x="{bx}" y="{y + 3}" width="{bw}" height="{c - 6}" rx="3" fill="#1b2a42"/>')
    gr = grow(t_mask + 0.5, 0.6, SCENE - .3)
    g.append(f'<rect class="{gr}" x="{bx}" y="{y + 3}" width="{bw * sc:.1f}" height="{c - 6}" rx="3" '
             f'fill="{FLAG if hit else NORM}"/>')
g.append(f'<line x1="{bx + bw * thr:.1f}" y1="{gy - 4}" x2="{bx + bw * thr:.1f}" y2="{gy + 6 * (c + 13) - 8}" '
         f'stroke="{TEXT}" stroke-opacity=".55" stroke-dasharray="3 3"/>')
g.append(text(bx + bw * thr, gy + 6 * (c + 13) + 8, "healthy noise floor", 11, MUTED, cls="mono", anchor="middle"))
g.append(f'<g class="{flagged_knee}">' + text(bx + bw + 8, gy + 4 * (c + 13) + 13, "flag", 13, FLAG, 700) + "</g>")
g.append(text(724, 404, "Each joint is hidden in turn and predicted from the others.", 15, "#c8d6ea"))
g.append(text(724, 426, "A joint is flagged when its error exceeds the healthy noise floor.", 15, "#c8d6ea"))
g.append("</g>")
body += g

# ---------------------------------------------------------------- scene 2
t0 = SCENE
s = show(t0 + .15, t0 + SCENE - .3, name="s1")
g = [f'<g class="{s}">', header(2, "RECONSTRUCT A NORMATIVE TWIN", "Flagged joints are rebuilt from the rest")]
px0, px1, py0, py1 = 780, 1210, 140, 340
ymax = 70


def X(p):
    return px0 + (px1 - px0) * p / 100


def Y(v):
    return py1 - (py1 - py0) * v / ymax


def knee_curve(p, swing_peak):
    """Illustrative knee flexion over one gait cycle (deg)."""
    stance = 16 * math.exp(-((p - 14) / 9) ** 2)
    swing = swing_peak * math.exp(-((p - 72) / 12) ** 2)
    return 4 + stance + swing


def path(f):
    pts = [f"{X(p):.1f},{Y(f(p)):.1f}" for p in range(0, 101, 2)]
    return "M " + " L ".join(pts)


up = [f"{X(p):.1f},{Y(knee_curve(p, 56) + 6):.1f}" for p in range(0, 101, 2)]
lo = [f"{X(p):.1f},{Y(max(knee_curve(p, 56) - 6, 0)):.1f}" for p in range(100, -1, -2)]
for v in (0, 20, 40, 60):
    g.append(f'<line x1="{px0}" y1="{Y(v):.1f}" x2="{px1}" y2="{Y(v):.1f}" stroke="{LINE}"/>')
    g.append(text(px0 - 10, Y(v) + 4, f"{v}°", 12, MUTED, cls="mono", anchor="end"))
for p in (0, 50, 100):
    g.append(text(X(p), py1 + 20, f"{p}%", 12, MUTED, cls="mono", anchor="middle"))
g.append(text((px0 + px1) / 2, py1 + 40, "gait cycle, knee flexion", 12, MUTED, cls="mono", anchor="middle"))
band = show(t0 + .5, t0 + SCENE - .3)
g.append(f'<g class="{band}"><path d="M {" L ".join(up)} L {" L ".join(lo)} Z" fill="{NORM}" fill-opacity=".16"/>'
         + text(X(24), Y(31), "normative range", 12, NORM, cls="mono") + "</g>")
d1 = draw(t0 + .9, 1.8, t0 + SCENE - .3)
g.append(f'<path class="{d1}" pathLength="1" d="{path(lambda p: knee_curve(p, 30))}" fill="none" '
         f'stroke="{FLAG}" stroke-width="3.5" stroke-linecap="round"/>')
d2 = draw(t0 + 3.0, 1.8, t0 + SCENE - .3)
g.append(f'<path class="{d2}" pathLength="1" d="{path(lambda p: knee_curve(p, 46))}" fill="none" '
         f'stroke="{TEXT}" stroke-width="3" stroke-dasharray="1" stroke-linecap="round"/>')
l1, l2 = show(t0 + 1.6, t0 + SCENE - .3), show(t0 + 3.8, t0 + SCENE - .3)
g.append(f'<g class="{l1}"><rect x="1030" y="128" width="14" height="4" rx="2" fill="{FLAG}"/>'
         + text(1050, 134, "paretic knee", 12, FLAG, cls="mono") + "</g>")
g.append(f'<g class="{l2}"><rect x="1030" y="146" width="14" height="4" rx="2" fill="{TEXT}"/>'
         + text(1050, 152, "reconstructed", 12, TEXT, cls="mono") + "</g>")
cap = show(t0 + 4.8, t0 + SCENE - .3)
g.append(f'<g class="{cap}">'
         + text(724, 404, "50 stroke survivors: reconstruction moved paretic hip and knee", 15, "#c8d6ea")
         + text(724, 426, "towards the normative mean in 8 of 8 comparisons (P &lt; 0.05 in 6).", 15, "#c8d6ea")
         + "</g>")
g.append(text(1228, 380, "illustrative curves", 11, MUTED, cls="mono", anchor="end"))
g.append("</g>")
body += g

# ---------------------------------------------------------------- scene 3
t0 = 2 * SCENE
s = show(t0 + .15, t0 + SCENE - .3, name="s2")
g = [f'<g class="{s}">', header(3, "HEALTHY ADULTS HELD OUT", "Angles kept, yet many trials flagged")]
# E1, reconstructed minus original RMSE, 90% CI, results/e1/README.md
rows = [("pelvis flexion", (0.03, -0.01, 0.08), (0.01, -0.02, 0.04)),
        ("R hip ab/adduction", (0.04, 0.02, 0.06), (0.00, -0.00, 0.01)),
        ("R hip flexion", (0.03, -0.00, 0.08), (0.00, -0.03, 0.03)),
        ("R knee flexion", (-0.57, -1.12, -0.12), (-0.01, -0.01, 0.00))]
fx0, fx1 = 760, 960
fy0 = 150


def FX(v):
    return fx0 + (fx1 - fx0) * (v + 2) / 4


g.append(f'<rect x="{FX(-1.5):.1f}" y="{fy0 - 26}" width="{FX(1.5) - FX(-1.5):.1f}" height="{4 * 50 - 4}" '
         f'fill="{RULE}" fill-opacity=".08"/>')
for v in (-1.5, 0, 1.5):
    g.append(f'<line x1="{FX(v):.1f}" y1="{fy0 - 26}" x2="{FX(v):.1f}" y2="{fy0 + 170}" stroke="{RULE if v else LINE}" '
             f'stroke-opacity="{.6 if v else 1}" stroke-dasharray="{"4 3" if v else "0"}"/>')
    g.append(text(FX(v), fy0 + 188, f"{v:+.1f}°" if v else "0", 12, MUTED, cls="mono", anchor="middle"))
g.append(text(FX(0), fy0 + 206, "change in RMSE, 90% CI", 12, MUTED, cls="mono", anchor="middle"))
for i, (name, a, b) in enumerate(rows):
    y = fy0 + i * 50
    g.append(text(fx0 - 36, y - 10, name, 12, "#c8d6ea", cls="mono"))
    for j, (m, lo_, hi_) in enumerate((a, b)):
        yy = y + 4 + j * 12
        on = t0 + 0.8 + 0.35 * i + 0.15 * j
        gr = grow(on, 0.5, t0 + SCENE - .3, origin="50% 50%")
        pp = pop(on + 0.2, t0 + SCENE - .3)
        col = NORM if j == 0 else "#b6c9e6"
        g.append(f'<line class="{gr}" x1="{FX(lo_):.1f}" y1="{yy}" x2="{FX(hi_) + .01:.1f}" y2="{yy}" '
                 f'stroke="{col}" stroke-width="2.5" stroke-linecap="round"/>')
        g.append(f'<circle class="{pp}" cx="{FX(m):.1f}" cy="{yy}" r="4.5" fill="{col}"/>')
g.append(text(fx0 - 36, fy0 - 40, "within ±1.5°: 4 of 4, both runs", 13, RULE, 700))

# trial grids: 48 trials, flagged counts from results/e3/e3c_README.md (7 frames)
tx0 = 1010
for gi, (lab, nflag, top) in enumerate([("run 1", 21, 140), ("run 2", 15, 278)]):
    order = sorted(range(48), key=lambda k: (k * 37 + gi * 11) % 48)[:nflag]
    g.append(text(tx0, top - 12, f"{lab}: {nflag} of 48 trials flagged ({round(100 * nflag / 48)}%)",
                  12, "#c8d6ea", cls="mono"))
    for k in range(48):
        x = tx0 + (k % 12) * 17
        y = top + (k // 12) * 17
        g.append(f'<rect x="{x}" y="{y}" width="13" height="13" rx="2.5" fill="#22334f"/>')
    for n_, k in enumerate(sorted(order)):
        x = tx0 + (k % 12) * 17
        y = top + (k // 12) * 17
        on = t0 + 1.8 + gi * 1.6 + 0.06 * n_
        pp = pop(on, t0 + SCENE - .3)
        g.append(f'<rect class="{pp}" x="{x}" y="{y}" width="13" height="13" rx="2.5" fill="{FLAG}"/>')
cap = show(t0 + 4.6, t0 + SCENE - .3)
g.append(f'<g class="{cap}">'
         + text(724, 404, "17 held-out adults: reconstruction kept the reported angles, but", 15, "#c8d6ea")
         + text(724, 426, "a third or more of trials still had a flagged joint.", 15, "#c8d6ea") + "</g>")
g.append("</g>")
body += g

# ---------------------------------------------------------------- scene 4
t0 = 3 * SCENE
s = show(t0 + .15, T - .35, name="s3")
g = [f'<g class="{s}">', header(4, "STROKE: WHICH SIDE IS PARETIC?", "A one-line clinical rule does better")]
ax0, ax1 = 930, 1210


def AX(v):
    return ax0 + (ax1 - ax0) * (v - 0.2) / 0.8


for v in (0.2, 0.4, 0.6, 0.8, 1.0):
    g.append(f'<line x1="{AX(v):.1f}" y1="124" x2="{AX(v):.1f}" y2="336" stroke="{LINE}"/>')
    g.append(text(AX(v), 354, f"{v:.1f}", 12, MUTED, cls="mono", anchor="middle"))
g.append(f'<line x1="{AX(.5):.1f}" y1="124" x2="{AX(.5):.1f}" y2="336" stroke="{TEXT}" stroke-opacity=".5" stroke-dasharray="4 3"/>')
g.append(text(AX(.5), 118, "chance", 11, MUTED, cls="mono", anchor="middle"))
g.append(text((ax0 + ax1) / 2, 372, "AUROC, paretic side, 95% CI", 12, MUTED, cls="mono", anchor="middle"))
# results/e2/README.md
items = [("model, run 1", (0.54, 0.38, 0.70), NORM, 160),
         ("model, run 2", (0.62, 0.46, 0.78), NORM, 205),
         ("knee flexion rule", (0.84, 0.70, 0.95), RULE, 250)]
for i, (lab, (m, lo_, hi_), col, y) in enumerate(items):
    on = t0 + 0.8 + 0.9 * i
    g.append(text(724, y + 5, lab, 14, col if i == 2 else "#c8d6ea", 700 if i == 2 else 400))
    gr = grow(on, 0.6, T - .35, origin="50% 50%")
    pp = pop(on + 0.3, T - .35)
    g.append(f'<line class="{gr}" x1="{AX(lo_):.1f}" y1="{y}" x2="{AX(hi_):.1f}" y2="{y}" stroke="{col}" '
             f'stroke-width="3" stroke-linecap="round"/>')
    g.append(f'<g class="{pp}"><circle cx="{AX(m):.1f}" cy="{y}" r="7" fill="{col}"/>'
             + text(AX(m), y - 13, f"{m:.2f}", 13, col, 700, cls="mono", anchor="middle") + "</g>")
g.append(text(724, 268, "lower peak in swing", 12, MUTED, cls="mono"))
# E3: mean AUROC of two runs per setting (e3a stroke_side.tsv, e3b, e3c, e3d READMEs)
e3 = [0.464, 0.474, 0.558, 0.529, 0.553, 0.35, 0.30, 0.66, 0.49, 0.41, 0.47]
e3lab = show(t0 + 3.6, T - .35)
g.append(f'<g class="{e3lab}">' + text(724, 305, "E3, other settings", 14, "#c8d6ea")
         + text(724, 322, "mean of two runs", 12, MUTED, cls="mono") + "</g>")
for i, v in enumerate(sorted(e3)):
    pp = pop(t0 + 3.8 + 0.12 * i, T - .35)
    g.append(f'<rect class="{pp}" x="{AX(v) - 2:.1f}" y="300" width="4" height="22" rx="2" fill="{SUB}" fill-opacity=".85"/>')
cap = show(t0 + 5.4, T - .35)
g.append(f'<g class="{cap}">'
         + text(724, 404, "No setting averaged above 0.66: training size, age range of the", 15, "#c8d6ea")
         + text(724, 426, "reference, window length (7, 15, 31 frames), ankles, peak statistic.", 15, "#c8d6ea")
         + "</g>")
g.append("</g>")
body += g

# ---------------------------------------------------------------- assemble
desc = ("Four scenes. 1: a walking figure; each of six joints (neck, pelvis, hips, knees) is withheld in turn "
        "from a 7-frame window and predicted from the others, and the knee, whose error exceeds the healthy noise "
        "floor, is flagged. 2: a paretic knee flexion curve is reconstructed towards the normative range; in 50 "
        "stroke survivors reconstruction moved paretic hip and knee towards the normative mean in 8 of 8 "
        "comparisons. 3: in 17 held-out healthy adults all four reported angles stayed within 1.5 degrees, yet 21 "
        "and 15 of 48 trials were flagged in the two runs. 4: the paretic side was identified with AUROC 0.54 and "
        "0.62 by the model and 0.84 by the side with lower peak knee flexion in swing; no other setting averaged "
        "above 0.66.")
style = "\n".join([
    '.sans{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}',
    '.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}',
    *css,
    "@media (prefers-reduced-motion: reduce){*{animation:none!important}"
    ".s0,.s0 *{opacity:1!important}.prog{display:none}}",
])
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 480" width="1280" height="480" role="img" aria-labelledby="t d">
  <title id="t">MotorPhen</title>
  <desc id="d">{desc}</desc>
  <style>
{style}
  </style>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{BG0}"/><stop offset=".55" stop-color="{BG1}"/><stop offset="1" stop-color="{BG2}"/></linearGradient>
    <radialGradient id="glow" cx="76%" cy="45%" r="45%"><stop offset="0" stop-color="{NORM}" stop-opacity=".14"/><stop offset="1" stop-color="{NORM}" stop-opacity="0"/></radialGradient>
    <clipPath id="card"><rect width="1280" height="480" rx="24"/></clipPath>
    <clipPath id="term"><rect x="92" y="300" width="560" height="54" rx="12"/></clipPath>
    <pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.1" fill="{NORM}" fill-opacity=".07"/></pattern>
  </defs>
  <g clip-path="url(#card)">
    <rect width="1280" height="480" fill="url(#bg)"/><rect width="1280" height="480" fill="url(#dots)"/><rect width="1280" height="480" fill="url(#glow)"/>
    <rect x="0" y="476" width="1280" height="4" fill="{LINE}"/>
    <rect class="prog" x="0" y="476" width="1280" height="4" fill="{FLAG}" fill-opacity=".7"/>
  </g>
  <rect x=".5" y=".5" width="1279" height="479" rx="24" fill="none" stroke="{LINE}"/>
  <rect x="700" y="30" width="552" height="420" rx="18" fill="#0a111d" fill-opacity=".55" stroke="{LINE}"/>
{chr(10).join("  " + b for b in body)}
</svg>
'''
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(svg)
print(f"wrote {OUT} ({len(svg):,} bytes)")
