"""
Bayesian precision — three-part explainer.

Scenes (render each separately):
  DataProcess  -> 00a_data_process.gif
  Inference    -> 00b_inference.gif
  Updating     -> 00c_updating.gif

Real photos: drop  cat.png / dog.png  into  animations/assets/
(falls back to placeholder cards otherwise).
"""

from pathlib import Path
from manim import *
import numpy as np
from scipy import stats
from scipy.special import comb as scipy_comb


# xelatex -> .xdv -> dvisvgm   (no Ghostscript needed)
config.tex_template = TexTemplate(
    tex_compiler="xelatex",
    output_format=".xdv",
    documentclass=r"\documentclass[preview]{standalone}",
    preamble=r"\usepackage{amsmath,amssymb}",
)
config.background_color = "#0D1117"

ASSETS  = Path(__file__).parent / "assets"
ANIMALS = Path(__file__).parent.parent / "media" / "animals"   # cat_1.png, dog_1.png, ...

PRIOR_C = "#5B9BD5"
LIK_C   = "#ED9B40"
POST_C  = "#5BC98C"
CI_C    = "#B58BE0"
YEL     = "#F5C542"
RED_C   = "#E06A6A"
GRAY_LT = "#8B919A"
INK     = "#E8EAED"


# ─── curve helpers ──────────────────────────────────────────────────────────
def _pts(ax, xs, ys):
    return [ax.c2p(x, y) for x, y in zip(xs, ys)]


def curve_from(ax, xs, ys, color, lw=3.2):
    return VMobject(color=color, stroke_width=lw).set_points_smoothly(_pts(ax, xs, ys))


def fill_from(ax, xs, ys, color, op=0.16):
    pts = [ax.c2p(xs[0], 0), *_pts(ax, xs, ys), ax.c2p(xs[-1], 0)]
    return Polygon(*pts, color=color, fill_opacity=op, stroke_width=0)


def beta_curve(ax, a, b, color, xs, lw=3.2):
    return curve_from(ax, xs, stats.beta(a, b).pdf(xs), color, lw)


def beta_fill(ax, a, b, color, xs, op=0.16):
    return fill_from(ax, xs, stats.beta(a, b).pdf(xs), color, op)


def nice_axis(x_range, y_range, x_len, y_len, hide_y=True):
    ax = Axes(
        x_range=x_range, y_range=y_range, x_length=x_len, y_length=y_len,
        axis_config={"color": GRAY_LT, "stroke_width": 2,
                     "include_numbers": False, "tick_size": 0.055},
        x_axis_config={"longer_tick_multiple": 1.0},
    )
    if hide_y:
        ax.y_axis.set_opacity(0)
    return ax


def x_ticklabels(ax, vals, fs=18):
    g = VGroup()
    for v in vals:
        g.add(Text(f"{v:.1f}", font_size=fs, color=GRAY_LT)
              .next_to(ax.c2p(v, 0), DOWN, buff=0.14))
    return g


def vline(ax, x, y_top, color=YEL, lw=2.2):
    return DashedLine(ax.c2p(x, 0), ax.c2p(x, y_top),
                      color=color, stroke_width=lw, dash_length=0.11)


def titled_formula(title_str, tex, color):
    title = Text(title_str, font_size=24, color=color, weight=BOLD)
    rule  = Line(ORIGIN, RIGHT, color=color, stroke_width=2).set_width(title.width * 1.25)
    rule.next_to(title, DOWN, buff=0.10)
    formula = MathTex(tex, font_size=32, color=INK).next_to(rule, DOWN, buff=0.22)
    return VGroup(title, rule, formula)


# ─── icons ──────────────────────────────────────────────────────────────────
def load_photo(kind, h):
    """Real image from assets/ (falls back to a label) sized to height h."""
    for ext in (".png", ".jpg", ".jpeg"):
        p = ASSETS / f"{kind}{ext}"
        if p.exists():
            im = ImageMobject(str(p))
            im.height = h
            return im
    return Text(kind, font_size=26, color=INK)


def nn_inside(width, height):
    """Neural-network glyph (edges + nodes only) to embed inside a box."""
    layers = [3, 4, 4, 2]
    xs = np.linspace(-width / 2, width / 2, len(layers))
    node_pos = []
    for li, n in enumerate(layers):
        ys = np.linspace(height / 2, -height / 2, n) if n > 1 else [0]
        node_pos.append([np.array([xs[li], y, 0]) for y in ys])
    edges = VGroup()
    for li in range(len(layers) - 1):
        for a in node_pos[li]:
            for b in node_pos[li + 1]:
                edges.add(Line(a, b, color=PRIOR_C, stroke_width=0.8).set_opacity(0.28))
    nodes = VGroup()
    for layer in node_pos:
        for p in layer:
            nodes.add(Dot(p, radius=0.05, color=INK).set_stroke(PRIOR_C, width=1.3))
    return VGroup(edges, nodes)


def speech_bubble(text, color=INK):
    t = Text(text, font_size=20, color=color)
    bubble = RoundedRectangle(width=t.width + 0.4, height=t.height + 0.34,
                              corner_radius=0.12, color=color, stroke_width=1.8,
                              fill_color="#0D1117", fill_opacity=1)
    t.move_to(bubble)
    # downward-pointing tail flush with the bubble's bottom edge (no overlap into text)
    tail = Triangle(color=color, fill_color="#0D1117", fill_opacity=1, stroke_width=1.8)
    tail.scale(0.11).rotate(PI)                       # apex points down
    tail.next_to(bubble, DOWN, buff=0.0).shift(LEFT * 0.3 + UP * 0.03)
    return VGroup(bubble, tail, t)


# ─── base scene with top captions ───────────────────────────────────────────
class Base(Scene):
    def setup(self):
        self._cap = None

    def say(self, text, size=26, run=0.5, hold=0.0):
        new = Text(text, font_size=size, color=INK).to_edge(UP, buff=0.45)
        if self._cap is not None:
            self.play(FadeOut(self._cap, run_time=run))
        self._cap = new
        self.play(FadeIn(new, shift=DOWN * 0.12, run_time=run))
        if hold:
            self.wait(hold)

    def cap_anims(self, text, size=26):
        """Return caption-swap animations to fold into a self.play() so the
        caption changes in lockstep with the visuals (no lead/lag)."""
        new = Text(text, font_size=size, color=INK).to_edge(UP, buff=0.45)
        anims = []
        if self._cap is not None:
            anims.append(FadeOut(self._cap))
        anims.append(FadeIn(new, shift=DOWN * 0.12))
        self._cap = new
        return anims

    def clear_cap(self):
        if self._cap is not None:
            self.play(FadeOut(self._cap, run_time=0.3))
            self._cap = None

    def pulse(self, line, color, rt=0.55):
        dot = Dot(line.get_start(), radius=0.075, color=color)
        glow = dot.copy().set_opacity(0.35).scale(2.2)
        glow.add_updater(lambda m: m.move_to(dot))
        self.add(glow, dot)
        self.play(MoveAlongPath(dot, line), rate_func=smooth, run_time=rt)
        self.remove(dot, glow)


# ════════════════════════════════════════════════════════════════════════════
# SCENE 1 — the data-generating process  ->  the precision estimate
# ════════════════════════════════════════════════════════════════════════════
class DataProcess(Base):
    def construct(self):
        self.say("Where does a precision number actually come from?", hold=1.4)

        # pipeline scaffold (identical boxes, evenly spaced) --------------
        BW, BH, BY = 2.0, 1.8, 0.5

        def make_box(cx):
            return RoundedRectangle(width=BW, height=BH, corner_radius=0.16,
                                    color=GRAY_LT, stroke_width=2.2).move_to(RIGHT * cx + UP * BY)

        img_box, model_box, human_box = make_box(-4.4), make_box(-1.1), make_box(2.2)

        # cycle through the distinct photos in media/animals (cat_1..n, dog_1..n)
        cat_files = sorted(ANIMALS.glob("cat_*.png"))
        dog_files = sorted(ANIMALS.glob("dog_*.png"))
        idx = {"cat": 0, "dog": 0}

        def next_photo(kind):
            files = cat_files if kind == "cat" else dog_files
            if not files:
                return load_photo(kind, BH * 0.78)
            f = files[idx[kind] % len(files)]
            idx[kind] += 1
            im = ImageMobject(str(f))
            im.height = BH * 0.78
            if im.width > BW * 0.86:        # keep landscape photos inside the box
                im.width = BW * 0.86
            return im

        photo = next_photo("cat").move_to(img_box)
        nn = nn_inside(BW * 0.72, BH * 0.46).move_to(model_box).shift(UP * 0.16)
        model_lbl = Text("model", font_size=20, color=INK, weight=BOLD).next_to(nn, DOWN, buff=0.16)
        model = VGroup(model_box, nn, model_lbl)
        human_lbl = Text("human\ncheck", font_size=22, color=INK, weight=BOLD,
                         line_spacing=0.8).move_to(human_box)
        human = VGroup(human_box, human_lbl)

        def conn(a, b):
            ln = Line(a, b, color=GRAY_LT, stroke_width=2.5).set_opacity(0.55)
            ln.add_tip(tip_shape=StealthTip, tip_length=0.2)
            ln.get_tip().set_opacity(0.7)
            return ln
        c1 = conn(img_box.get_right() + RIGHT*0.05, model_box.get_left() - RIGHT*0.05)
        c2 = conn(model_box.get_right() + RIGHT*0.05, human_box.get_left() - RIGHT*0.05)

        tp_lbl = Text("TP = 0", font_size=30, color=POST_C).move_to(RIGHT * 5.3 + UP * 1.0)
        fp_lbl = Text("FP = 0", font_size=30, color=RED_C ).move_to(RIGHT * 5.3 + UP * 0.1)
        c3 = conn(human_box.get_right() + RIGHT*0.05, RIGHT * 4.4 + UP * BY)

        self.play(FadeIn(img_box), FadeIn(photo, shift=RIGHT*0.2), run_time=0.7)
        self.play(Create(c1), FadeIn(model, shift=RIGHT*0.2), run_time=0.8)
        self.say("A model looks at each image and predicts a label.", hold=0.7)
        self.play(Create(c2), FadeIn(human, shift=RIGHT*0.2), run_time=0.7)
        self.play(Create(c3), FadeIn(tp_lbl), FadeIn(fp_lbl), run_time=0.6)
        self.say("A human annotator then checks each prediction:  is it really a cat?", hold=0.7)

        # ── precision formula at the bottom (fills in live) ──────────────
        phat = MathTex(
            r"\hat{p}", r"=", r"\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}",
            font_size=46,
        ).move_to(DOWN * 2.5)
        phat[0].set_color(YEL)
        self.play(Write(phat), run_time=0.8)

        tp = fp = 0

        def swap_photo(kind, fast=False):
            nonlocal photo
            newp = next_photo(kind).move_to(img_box)
            if fast:
                self.play(FadeOut(photo, run_time=0.08))
                photo = newp
                self.play(FadeIn(photo, run_time=0.14))
            else:
                self.play(FadeOut(photo, run_time=0.22))
                photo = newp
                self.play(FadeIn(photo, run_time=0.45))

        # the model always predicts "cat" — show these overhead bubbles once, keep them
        pred = speech_bubble("prediction:  cat", color=PRIOR_C).next_to(model_box, UP, buff=0.4)
        q    = speech_bubble("really a cat?",   color=INK).next_to(human_box, UP, buff=0.4)
        bubbles_shown = [False]

        def detailed(kind, correct):
            nonlocal tp, fp
            swap_photo(kind)
            self.pulse(c1, YEL)
            if not bubbles_shown[0]:
                self.play(FadeIn(pred, shift=UP*0.1), run_time=0.4)
            self.pulse(c2, YEL)
            if not bubbles_shown[0]:
                self.play(FadeIn(q, shift=UP*0.1), run_time=0.4)
                bubbles_shown[0] = True
            self.wait(0.4)
            self.pulse(c3, POST_C if correct else RED_C)

            if correct:
                tag = Text("yes  →  TP", font_size=20, color=POST_C).next_to(human_box, DOWN, buff=0.28)
                tp += 1
                nt = Text(f"TP = {tp}", font_size=30, color=POST_C).move_to(tp_lbl)
                self.play(FadeIn(tag), run_time=0.35)
                self.play(Transform(tp_lbl, nt),
                          Flash(tp_lbl, color=POST_C, line_length=0.22, num_lines=12), run_time=0.6)
            else:
                tag = Text("no (a dog)  →  FP", font_size=20, color=RED_C).next_to(human_box, DOWN, buff=0.28)
                fp += 1
                nf = Text(f"FP = {fp}", font_size=30, color=RED_C).move_to(fp_lbl)
                self.play(FadeIn(tag), run_time=0.35)
                self.play(Transform(fp_lbl, nf),
                          Flash(fp_lbl, color=RED_C, line_length=0.22, num_lines=12), run_time=0.6)
            self.wait(0.6)
            self.play(FadeOut(tag), run_time=0.35)   # keep pred & q; only the verdict tag is transient

        self.say("A real cat, predicted cat — correct.  A true positive (TP).")
        detailed("cat", True)
        self.say("A dog, also predicted cat — wrong.  A false positive (FP).")
        detailed("dog", False)

        # fast montage to TP=7, FP=3 -------------------------------------
        self.say("Run it over the whole test set.")
        montage = [True, True, False, True, True, True, True, False]
        for ok in montage:
            swap_photo("cat" if ok else "dog", fast=True)
            col = POST_C if ok else RED_C
            if ok:
                tp += 1; lbl = Text(f"TP = {tp}", font_size=30, color=POST_C).move_to(tp_lbl)
                self.play(c1.animate.set_color(col), c2.animate.set_color(col), c3.animate.set_color(col),
                          Transform(tp_lbl, lbl), run_time=0.2)
            else:
                fp += 1; lbl = Text(f"FP = {fp}", font_size=30, color=RED_C).move_to(fp_lbl)
                self.play(c1.animate.set_color(col), c2.animate.set_color(col), c3.animate.set_color(col),
                          Transform(fp_lbl, lbl), run_time=0.2)
            self.play(c1.animate.set_color(GRAY_LT), c2.animate.set_color(GRAY_LT),
                      c3.animate.set_color(GRAY_LT), run_time=0.1)

        self.wait(0.3)

        # ── reveal the precision estimate ───────────────────────────────
        self.say("After 10 checks: 7 correct, 3 wrong.  Plug them in:")
        phat_full = MathTex(
            r"\hat{p}", r"=", r"\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}",
            r"=", r"\frac{7}{7+3}", r"=", r"0.70",
            font_size=46,
        ).move_to(DOWN * 2.5)
        phat_full[0].set_color(YEL)
        phat_full[6].set_color(YEL)
        self.play(TransformMatchingTex(phat, phat_full), run_time=1.0)
        self.play(Flash(phat_full[6], color=YEL, line_length=0.25, num_lines=14), run_time=0.7)
        self.wait(1.0)

        # ── combine: p-hat estimate  ->  the true coin-rate p ───────────
        self.say("But 0.70 is just an estimate.  What is the true precision p?", hold=0.8)
        self.play(FadeOut(Group(img_box, photo, model, human, c1, c2, c3, tp_lbl, fp_lbl, pred, q)), run_time=0.5)
        self.play(phat_full.animate.scale(0.78).move_to(LEFT * 3.0 + UP * 0.7), run_time=0.6)

        # the yellow rounded true-precision p (a biased coin), same line as p-hat
        coin = Circle(radius=0.72, color=YEL, fill_opacity=0.12, stroke_width=3).move_to(RIGHT * 3.6 + UP * 0.7)
        coin_p = MathTex(r"p", font_size=58, color=YEL).move_to(coin)
        self.play(FadeIn(coin, scale=0.85), Write(coin_p), run_time=0.7)

        # dashed arrow: estimate -> truth, horizontal
        arrow = DashedLine(phat_full.get_right() + RIGHT * 0.2, coin.get_left() + LEFT * 0.12,
                           color=GRAY_LT, stroke_width=2.2, dash_length=0.12)
        arrow.add_tip(tip_shape=StealthTip, tip_length=0.18)
        est = Text("estimates", font_size=22, color=GRAY_LT).next_to(arrow, UP, buff=0.12)
        self.play(Create(arrow), FadeIn(est), run_time=0.6)

        self.say("Each prediction is one flip of this biased coin with rate p.")
        bern = MathTex(r"X_i \sim \mathrm{Bernoulli}(p)", font_size=44, color=INK).move_to(DOWN * 1.5)
        sub  = MathTex(r"p\ \text{is fixed, but unknown}", font_size=30, color=GRAY_LT).next_to(bern, DOWN, buff=0.35)
        self.play(Write(bern), run_time=0.7)
        self.play(FadeIn(sub, shift=UP*0.12), run_time=0.6)
        self.wait(1.4)
        self.say("So instead of one number, we will estimate a full distribution for p.", hold=1.6)
        self.clear_cap()
        self.play(*[FadeOut(o) for o in self.mobjects], run_time=0.6)


# ════════════════════════════════════════════════════════════════════════════
# SCENE 2 — prior x likelihood = posterior
# ════════════════════════════════════════════════════════════════════════════
class Inference(Base):
    def construct(self):
        xs = np.linspace(0.001, 0.999, 500)
        TP, FP = 7, 3

        ax = nice_axis([0, 1, 0.2], [0, 6, 2], 9.8, 2.9).to_edge(DOWN, buff=1.05)
        xt = x_ticklabels(ax, [0, .2, .4, .6, .8, 1.0])
        xl = MathTex(r"p \;\;(\text{true precision})", font_size=22, color=GRAY_LT).next_to(xt, DOWN, buff=0.18)
        self.play(Create(ax), FadeIn(xt), FadeIn(xl), run_time=0.8)

        # likelihood ------------------------------------------------------
        lik = scipy_comb(TP + FP, TP) * xs**TP * (1 - xs)**FP
        lik = lik / lik.max() * 3.0
        lik_c = curve_from(ax, xs, lik, LIK_C)
        lik_f = fill_from(ax, xs, lik, LIK_C, 0.13)
        lik_block = titled_formula(
            "Likelihood",
            r"\mathbb{P}(\text{data}\mid p)\propto p^{\mathrm{TP}}(1-p)^{\mathrm{FP}}",
            LIK_C,
        ).to_corner(UR, buff=0.55).shift(DOWN * 0.85)

        self.say("The data alone — 7 correct, 3 wrong — gives the likelihood of p.")
        self.play(Create(lik_c), FadeIn(lik_f), run_time=1.0)
        self.play(FadeIn(lik_block, shift=DOWN * 0.1), run_time=0.7)
        self.say("It peaks near 0.70, but stays wide: 10 samples carry little certainty.", hold=1.6)

        # prior -----------------------------------------------------------
        pri_c = beta_curve(ax, 1, 1, PRIOR_C, xs)
        pri_f = beta_fill(ax, 1, 1, PRIOR_C, xs)
        pri_block = titled_formula(
            "Prior",
            r"\mathbb{P}(p)\propto p^{\alpha-1}(1-p)^{\beta-1}",
            PRIOR_C,
        ).to_corner(UL, buff=0.55).shift(DOWN * 0.85)

        self.say("Before the data, we state a prior belief.  Flat here = no preference.")
        self.play(Create(pri_c), FadeIn(pri_f), run_time=1.0)
        self.play(FadeIn(pri_block, shift=DOWN * 0.1), run_time=0.7)
        self.wait(1.4)

        # derivation ------------------------------------------------------
        self.say("Bayes' rule: posterior is prior times likelihood, then renormalised.", hold=0.5)
        lines = [
            MathTex(r"\mathbb{P}(p\mid\text{data})\propto\mathbb{P}(p)\cdot\mathbb{P}(\text{data}\mid p)",
                    font_size=32, color=INK),
            MathTex(r"\propto\,", r"p^{\alpha-1}(1-p)^{\beta-1}", r"\,\cdot\,",
                    r"p^{\mathrm{TP}}(1-p)^{\mathrm{FP}}", font_size=32),
            MathTex(r"=\,p^{(\alpha+\mathrm{TP})-1}(1-p)^{(\beta+\mathrm{FP})-1}",
                    font_size=34, color=POST_C),
        ]
        lines[1][1].set_color(PRIOR_C); lines[1][3].set_color(LIK_C)
        for ln in lines:
            ln.move_to(UP * 0.95)

        self.play(FadeIn(lines[0], shift=UP * 0.12), run_time=0.7); self.wait(1.3)
        self.play(FadeOut(lines[0], shift=UP * 0.12), run_time=0.3)
        self.play(FadeIn(lines[1], shift=UP * 0.12), run_time=0.7)
        self.say("Same base, so the exponents simply add.", hold=1.3)
        self.play(FadeOut(lines[1], shift=UP * 0.12), run_time=0.3)
        self.play(FadeIn(lines[2], shift=UP * 0.12), run_time=0.7); self.wait(1.2)

        boxed = MathTex(r"\mathbb{P}(p\mid\text{data})=\mathrm{Beta}(\alpha+\mathrm{TP},\,\beta+\mathrm{FP})",
                        font_size=34, color=POST_C).move_to(UP * 0.95)
        box = SurroundingRectangle(boxed, color=POST_C, buff=0.2, corner_radius=0.1)
        self.play(FadeOut(lines[2], shift=UP * 0.12), run_time=0.3)
        self.play(FadeIn(boxed), Create(box), run_time=0.8)
        self.say("The posterior is itself a Beta distribution — that is conjugacy.", hold=1.6)

        # keep a green Posterior formula between Prior (left) and Likelihood (right)
        post_block = titled_formula(
            "Posterior",
            r"\mathrm{Beta}(\alpha+\mathrm{TP},\,\beta+\mathrm{FP})",
            POST_C,
        ).to_edge(UP, buff=0.55).shift(DOWN * 0.85)

        # posterior curve -------------------------------------------------
        post_c = beta_curve(ax, 1 + TP, 1 + FP, POST_C, xs, lw=3.6)
        post_f = beta_fill(ax, 1 + TP, 1 + FP, POST_C, xs, 0.22)
        self.play(FadeOut(box, run_time=0.3),
                  ReplacementTransform(boxed, post_block[2]),
                  FadeIn(post_block[0]), FadeIn(post_block[1]), run_time=0.7)
        self.play(Create(post_c), FadeIn(post_f),
                  pri_c.animate.set_opacity(0.4), pri_f.animate.set_opacity(0.05),
                  lik_c.animate.set_opacity(0.4), lik_f.animate.set_opacity(0.05),
                  run_time=1.2)
        self.say("Uniform prior + data  →  Beta(8, 4): the green posterior.", hold=1.0)

        # posterior mode = the estimated precision -----------------------
        def mode_of(a, b):
            return (a - 1) / (a + b - 2)

        def mode_line(a, b, color=YEL):
            m = mode_of(a, b)
            return vline(ax, m, stats.beta(a, b).pdf(m), color, lw=2.2), m

        def status(txt, m, col):
            return VGroup(
                Text(txt, font_size=22, color=col),
                MathTex(rf"\hat{{p}}\ (\text{{mode}}) = {m:.2f}", font_size=26, color=YEL),
            ).arrange(DOWN, buff=0.14).move_to(UP * 1.0)

        mln, m0 = mode_line(1 + TP, 1 + FP)
        st = status("uniform prior  Beta(1,1)", m0, PRIOR_C)
        self.say("Its peak — the mode — is exactly the estimated precision  p̂ = TP/(TP+FP).")
        self.play(Create(mln), FadeIn(st), run_time=0.7)
        self.wait(1.2)

        # prior influence -------------------------------------------------
        self.say("And the prior shifts that estimate.  Watch the mode move.", hold=0.6)
        cases = [
            ((6, 2), "optimistic prior  Beta(6,2)", "An optimistic prior believes p is high  →  mode pulled right."),
            ((2, 6), "pessimistic prior  Beta(2,6)", "A pessimistic prior believes p is low  →  mode pulled left."),
        ]
        for (a, b), ptxt, expl in cases:
            ap, bp = a + TP, b + FP
            new_mln, m = mode_line(ap, bp)
            new_st = status(ptxt, m, PRIOR_C)
            self.play(
                Transform(pri_c, beta_curve(ax, a, b, PRIOR_C, xs)),
                Transform(pri_f, beta_fill(ax, a, b, PRIOR_C, xs, 0.05)),
                Transform(post_c, beta_curve(ax, ap, bp, POST_C, xs, lw=3.6)),
                Transform(post_f, beta_fill(ax, ap, bp, POST_C, xs, 0.22)),
                Transform(mln, new_mln),
                FadeOut(st), FadeIn(new_st),
                *self.cap_anims(expl),
                run_time=1.3,
            )
            st = new_st
            self.wait(1.4)

        self.say("With more data the likelihood sharpens and the prior's pull fades.", hold=1.6)
        self.clear_cap()
        self.play(*[FadeOut(o) for o in self.mobjects], run_time=0.6)


# ════════════════════════════════════════════════════════════════════════════
# SCENE 3 — sequential updating, ending on the estimate + 95% CI
# ════════════════════════════════════════════════════════════════════════════
class Updating(Base):
    def construct(self):
        xs = np.linspace(0.0005, 0.9995, 700)
        a0 = b0 = 1.0
        tp = fp = 0
        N = 160
        rng = np.random.default_rng(7)
        obs = list(rng.choice([1, 0], size=N, p=[0.76, 0.24]))

        PEAK = 0.92  # displayed peak height (curve normalised to fill frame)
        ax = nice_axis([0, 1, 0.2], [0, 1.05, 1], 10.2, 4.3).to_edge(DOWN, buff=0.8)
        xt = x_ticklabels(ax, [0, .2, .4, .6, .8, 1.0])
        xl = MathTex(r"p", font_size=26, color=GRAY_LT).next_to(ax, DOWN, buff=0.5)
        self.play(FadeIn(ax), FadeIn(xt), FadeIn(xl), run_time=0.6)

        def scaled(a, b):
            ys = stats.beta(a, b).pdf(xs)
            return ys / ys.max() * PEAK

        # counter sits just under the caption band (no persistent title -> no overlap)
        def counter(tp_, fp_):
            return MathTex(rf"\mathrm{{TP}}={tp_},\;\;\mathrm{{FP}}={fp_}",
                           font_size=28, color=GRAY_LT).move_to(UP * 2.95)

        # live mode line (= estimated precision p̂) and its value label
        def mode_marker(tp_, fp_):
            m = tp_ / (tp_ + fp_)
            ln = vline(ax, m, PEAK, YEL, lw=2.2)
            lbl = MathTex(rf"\hat{{p}}={m:.2f}", font_size=26, color=YEL).move_to(ax.c2p(m, PEAK) + UP * 0.26)
            return ln, lbl

        cnt   = counter(0, 0)
        curve = curve_from(ax, xs, scaled(a0, b0), POST_C, lw=3.4)
        fill  = fill_from(ax, xs, scaled(a0, b0), POST_C, 0.18)
        self.add(cnt)
        self.play(Create(curve), FadeIn(fill), run_time=0.7)
        self.say("Start from a flat prior — total uncertainty about p.", hold=1.0)

        # compact dot strip (samples), well above the curve
        grid0 = LEFT * 4.7 + UP * 2.3

        def rt(i):
            if i < 8:    return 0.5
            elif i < 22: return 0.24
            else:        return 0.05

        ml = mlbl = None
        for i, o in enumerate(obs):
            tp += o; fp += 1 - o
            a, b = a0 + tp, b0 + fp
            nc = curve_from(ax, xs, scaled(a, b), POST_C, lw=3.4)
            nf = fill_from(ax, xs, scaled(a, b), POST_C, 0.18)
            ncn = counter(tp, fp)
            new_ml, new_mlbl = mode_marker(tp, fp)
            dot = Dot(grid0 + RIGHT * (i % 32) * 0.30 + DOWN * (i // 32) * 0.26,
                      radius=0.065, color=POST_C if o else RED_C)
            anims = [Transform(curve, nc), Transform(fill, nf),
                     Transform(cnt, ncn), FadeIn(dot, scale=0.5)]
            if ml is None:
                ml, mlbl = new_ml, new_mlbl
                anims += [Create(ml), FadeIn(mlbl)]
            else:
                anims += [Transform(ml, new_ml), Transform(mlbl, new_mlbl)]
            self.play(*anims, run_time=rt(i))
            if i == 0:
                self.say("Each observation nudges the curve; the dotted line marks the live estimate p̂.")
            elif i == 21:
                self.say("With more data the posterior concentrates around the true rate.")
            elif i == N - 1:
                self.say("160 observations later: a sharp, confident posterior.")

        # ── 95% credible interval, smooth reveal ────────────────────────
        a, b = a0 + tp, b0 + fp
        dist = stats.beta(a, b)
        lo, hi = dist.interval(0.95)
        mean = dist.mean()

        mask = (xs >= lo) & (xs <= hi)
        xseg = xs[mask]
        yseg = scaled(a, b)[mask]
        ci_pts = [ax.c2p(xseg[0], 0), *_pts(ax, xseg, yseg), ax.c2p(xseg[-1], 0)]
        ci_band = Polygon(*ci_pts, color=CI_C, fill_opacity=0.0, stroke_width=0)
        lo_ln = vline(ax, lo, PEAK * 0.96, CI_C, lw=2)
        hi_ln = vline(ax, hi, PEAK * 0.96, CI_C, lw=2)

        self.say("The shaded band holds 95% of the posterior — our credible interval.")
        self.play(Create(lo_ln), Create(hi_ln), run_time=0.6)
        self.play(ci_band.animate.set_fill(CI_C, opacity=0.32), run_time=0.8)

        ci_txt = MathTex(rf"95\%\ \text{{CI}}=[{lo:.2f},\,{hi:.2f}]",
                         font_size=30, color=CI_C).move_to(LEFT * 3.7 + UP * 0.3)
        self.play(Write(ci_txt), run_time=0.7)
        self.wait(1.0)

        # ── final centred estimate ──────────────────────────────────────
        self.clear_cap()
        phat = VGroup(
            MathTex(r"\hat{p}", font_size=120, color=POST_C),
            MathTex(rf"= {mean:.2f}", font_size=120, color=INK),
        ).arrange(RIGHT, buff=0.3).move_to(UP * 0.4)
        sub = MathTex(rf"\text{{precision estimate}}\quad 95\%\ \text{{CI}}=[{lo:.2f},\,{hi:.2f}]",
                      font_size=34, color=GRAY_LT).next_to(phat, DOWN, buff=0.5)

        self.play(
            FadeOut(VGroup(curve, fill, ci_band, lo_ln, hi_ln, ci_txt, cnt, ml, mlbl)),
            FadeOut(ax), FadeOut(xt), FadeOut(xl),
            *[FadeOut(m) for m in self.mobjects
              if isinstance(m, Dot)],
            run_time=0.7,
        )
        self.play(Write(phat), run_time=1.0)
        self.play(FadeIn(sub, shift=UP * 0.15), run_time=0.7)
        self.wait(2.4)


# ════════════════════════════════════════════════════════════════════════════
# SCENE 4 — compare_models: Monte Carlo estimate of P(A > B)
# ════════════════════════════════════════════════════════════════════════════
class CompareModels(Base):
    def construct(self):
        # two model posteriors (precision)
        Aa, Ab = 1 + 80, 1 + 20      # Beta(81, 21)
        Ba, Bb = 1 + 72, 1 + 26      # Beta(73, 27)
        dA, dB = stats.beta(Aa, Ab), stats.beta(Ba, Bb)

        rng = np.random.default_rng(0)
        N = 6000
        a_s = dA.rvs(N, random_state=rng)
        b_s = dB.rvs(N, random_state=rng)
        run_est = np.cumsum((a_s > b_s).astype(float)) / np.arange(1, N + 1)
        P_true = float((dA.rvs(200_000, random_state=rng) > dB.rvs(200_000, random_state=rng)).mean())

        xs = np.linspace(0.55, 0.95, 400)

        # ── top panel: the two posteriors ───────────────────────────────
        tax = nice_axis([0.55, 0.95, 0.1], [0, 12, 4], 7.4, 1.7).move_to(UP * 1.75)
        tlabs = VGroup(*[Text(f"{v:.1f}", font_size=16, color=GRAY_LT)
                         .next_to(tax.c2p(v, 0), DOWN, buff=0.12) for v in [0.6, 0.7, 0.8, 0.9]])
        txl = Text("precision", font_size=18, color=GRAY_LT).next_to(tax, DOWN, buff=0.42)
        cA, fA = beta_curve(tax, Aa, Ab, POST_C, xs), beta_fill(tax, Aa, Ab, POST_C, xs, 0.14)
        cB, fB = beta_curve(tax, Ba, Bb, PRIOR_C, xs), beta_fill(tax, Ba, Bb, PRIOR_C, xs, 0.14)
        lblA = Text("Model A", font_size=22, color=POST_C).move_to(tax.c2p(0.86, 11.4))
        lblB = Text("Model B", font_size=22, color=PRIOR_C).move_to(tax.c2p(0.63, 11.4))

        self.say("Is Model A really better than Model B?")
        self.play(Create(tax), FadeIn(tlabs), FadeIn(txl), run_time=0.7)
        self.play(Create(cB), FadeIn(fB), Write(lblB), run_time=0.7)
        self.play(Create(cA), FadeIn(fA), Write(lblA), run_time=0.7)
        self.say("Their posteriors overlap — point estimates alone can't answer that.", hold=1.2)

        # ── bottom panel: convergence of the running estimate ───────────
        bax = nice_axis([0, 3.8, 1], [0, 1, 0.5], 7.4, 1.9, hide_y=False).move_to(DOWN * 2.1)
        bax.y_axis.set_opacity(1)
        ylabs = VGroup(*[Text(f"{v:.1f}", font_size=16, color=GRAY_LT)
                         .next_to(bax.c2p(0, v), LEFT, buff=0.12) for v in [0.0, 0.5, 1.0]])
        xlabs = VGroup(*[Text(t, font_size=15, color=GRAY_LT).next_to(bax.c2p(e, 0), DOWN, buff=0.12)
                         for e, t in [(0, "1"), (1, "10"), (2, "100"), (3, "1000")]])
        bxl = Text("number of samples (log scale)", font_size=18, color=GRAY_LT).next_to(bax, DOWN, buff=0.4)
        true_ln = DashedLine(bax.c2p(0, P_true), bax.c2p(3.8, P_true), color=GRAY_LT,
                             stroke_width=1.8, dash_length=0.1)
        true_lbl = MathTex(rf"P_{{\text{{true}}}}\approx {P_true:.2f}", font_size=22, color=GRAY_LT)
        true_lbl.next_to(bax.c2p(3.8, P_true), UR, buff=0.05).shift(LEFT * 0.9)

        self.say("Monte Carlo: draw one sample from each posterior, check if A > B, repeat.")
        self.play(Create(bax), FadeIn(ylabs), FadeIn(xlabs), FadeIn(bxl),
                  Create(true_ln), FadeIn(true_lbl), run_time=0.9)

        # ── readout + helpers ───────────────────────────────────────────
        def Pnum(est):
            return MathTex(rf"\mathbb{{P}}(A > B)\ \approx\ {est:.2f}",
                           font_size=40, color=YEL).move_to(DOWN * 0.5)

        def sample_ticks(n):
            a, b = a_s[n - 1], b_s[n - 1]
            ya, yb = dA.pdf(a), dB.pdf(b)
            g = VGroup(
                Line(tax.c2p(a, 0), tax.c2p(a, ya), color=POST_C, stroke_width=3),
                Dot(tax.c2p(a, ya), radius=0.055, color=POST_C),
                Line(tax.c2p(b, 0), tax.c2p(b, yb), color=PRIOR_C, stroke_width=3),
                Dot(tax.c2p(b, yb), radius=0.055, color=PRIOR_C),
            )
            return g

        # sample counts to plot (dense early, geometric later)
        S = sorted(set([i for i in range(1, 13)] +
                       [int(round(x)) for x in np.geomspace(13, N, 44)]))

        def rt(k):
            if k < 12:   return 0.32
            elif k < 26: return 0.15
            else:        return 0.06

        self.say("Each draw is a coin flip: did A beat B this time?")
        pts, line, num, tk = [], None, None, None
        for k, n in enumerate(S):
            pts.append(bax.c2p(np.log10(n), run_est[n - 1]))
            new_line = VMobject(color=YEL, stroke_width=3)
            new_line.set_points_as_corners(pts if len(pts) > 1 else [pts[0], pts[0]])
            new_num = Pnum(run_est[n - 1])
            new_tk = sample_ticks(n)
            anims = []
            if line is None:
                line, num, tk = new_line, new_num, new_tk
                anims = [Create(line), FadeIn(num), FadeIn(tk)]
            else:
                anims = [Transform(line, new_line), Transform(num, new_num), Transform(tk, new_tk)]
            self.play(*anims, run_time=rt(k))
            if k == 0:
                self.say("A green and a blue draw; A > B counts as a win.")
            elif k == 12:
                self.say("Speed up — thousands of draws.  The win-rate is settling.")
            elif k == len(S) - 1:
                self.say("It converges to a precise, stable estimate of P(A > B).")

        self.wait(0.5)

        # ── highlight the winner ────────────────────────────────────────
        self.play(FadeOut(tk), run_time=0.3)
        verdict = MathTex(rf"\mathbb{{P}}(A > B)\ \approx\ {run_est[-1]:.2f}",
                          font_size=44, color=POST_C).move_to(DOWN * 0.5)
        self.play(Transform(num, verdict),
                  cB.animate.set_opacity(0.25), fB.animate.set_opacity(0.04),
                  lblB.animate.set_opacity(0.4), run_time=0.8)
        self.say("Model A is better with about 86% probability — uncertainty included.", hold=2.0)
        self.clear_cap()
        self.play(*[FadeOut(o) for o in self.mobjects], run_time=0.6)


# ════════════════════════════════════════════════════════════════════════════
# SCENE 5 — prob_above_threshold: acceptance probability
# ════════════════════════════════════════════════════════════════════════════
class AcceptanceProbability(Base):
    def construct(self):
        # --- Intro Slide ---
        intro_title = Text("Acceptance Probability", font_size=40, color=INK).move_to(UP * 1.5)
        intro_formula = MathTex(r"\mathbb{P}(p > \tau)", font_size=72, color=YEL).move_to(ORIGIN)
        intro_desc = Text("We use a 0.7 example threshold for display (τ = 0.7).", font_size=28, color=GRAY_LT).move_to(DOWN * 1.5)

        self.play(FadeIn(intro_title), Write(intro_formula), run_time=1.0)
        self.play(FadeIn(intro_desc), run_time=0.8)
        self.wait(2.2)
        self.play(FadeOut(intro_title), FadeOut(intro_formula), FadeOut(intro_desc), run_time=0.6)

        xs = np.linspace(0.30, 0.9995, 500)
        PEAK = 7.0
        ax = nice_axis([0.3, 1.0, 0.1], [0, 9, 3], 9.6, 3.1).to_edge(DOWN, buff=0.9)
        xt = VGroup(*[Text(f"{v:.1f}", font_size=18, color=GRAY_LT).next_to(ax.c2p(v, 0), DOWN, buff=0.14)
                      for v in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
        xl = Text("precision", font_size=20, color=GRAY_LT).next_to(ax, DOWN, buff=0.45)
        self.play(Create(ax), FadeIn(xt), FadeIn(xl), run_time=0.7)

        def disp(a, b):
            full = stats.beta(a, b).pdf(xs)
            fac = PEAK / full.max()
            return full * fac, fac

        def shade(a, b, t, fac):
            seg = xs[xs >= t]
            ys = stats.beta(a, b).pdf(seg) * fac
            pts = [ax.c2p(seg[0], 0), *[ax.c2p(x, y) for x, y in zip(seg, ys)], ax.c2p(seg[-1], 0)]
            return Polygon(*pts, color=POST_C, fill_opacity=0.5, stroke_width=0)

        def readout(a, b):                            # acceptance bar fixed at 0.7
            P = float(stats.beta(a, b).sf(0.7))
            ro = MathTex(rf"\mathbb{{P}}(p > \tau) = {P*100:.0f}\%",
                         font_size=32, color=INK).move_to(UP * 1.6)
            return ro, P

        def badge(P):
            if P >= 0.80:    txt, c = "SATISFYING", POST_C
            elif P >= 0.70:  txt, c = "BORDERLINE", YEL
            elif P >= 0.20:  txt, c = "UNCERTAIN", GRAY_LT
            else:            txt, c = "INSUFFICIENT", RED_C
            b = Text(txt, font_size=26, color=c, weight=BOLD)
            box = SurroundingRectangle(b, color=c, buff=0.14, corner_radius=0.1)
            return VGroup(box, b).move_to(UP * 0.95)

        def make_header(n, rate):                     # big, prominent n & rate
            return VGroup(
                MathTex(rf"n = {n}", font_size=44, color=INK),
                MathTex(rf"\text{{rate}} = {rate}\%", font_size=44, color=PRIOR_C),
            ).arrange(RIGHT, buff=0.9).move_to(UP * 2.6)

        def build(tp, fp, n, rate):
            a, b = 1 + tp, 1 + fp
            ys, fac = disp(a, b)
            cv = curve_from(ax, xs, ys, PRIOR_C, lw=3.4)
            fl = fill_from(ax, xs, ys, PRIOR_C, 0.07)
            sh = shade(a, b, 0.7, fac)
            ro, P = readout(a, b)
            return cv, fl, sh, ro, badge(P), make_header(n, rate)

        # fixed acceptance bar at 0.7 (drawn once, stays the whole scene)
        thr = vline(ax, 0.7, PEAK, YEL, lw=2.4)
        thr_lbl = MathTex(r"\tau = 0.7", font_size=26, color=YEL).next_to(ax.c2p(0.7, PEAK), UP, buff=0.12)

        # scenarios — vary the sample size (rate fixed), then vary the rate (n fixed)
        seqN = [
            (7, 3, 10, 73,   "Only 10 samples — the verdict is still uncertain."),
            (37, 13, 50, 73,  "50 samples — borderline; a little more data could tip it."),
            (73, 27, 100, 73, "100 samples — now satisfying, above the 80% line."),
            (365, 135, 500, 73, "500 samples — comfortably satisfying."),
        ]
        seqR = [
            (65, 35, 100, 65, "A 65% rate is insufficient — below 20%."),
            (70, 30, 100, 70, "70% lands on the bar — uncertain."),
            (75, 25, 100, 75, "75% is satisfying at 100 samples."),
            (85, 15, 100, 85, "85% clears it decisively — satisfying."),
        ]

        self.say("Will the true precision clear the acceptance threshold \\tau?")
        self.play(Create(thr), FadeIn(thr_lbl), run_time=0.6)

        cv, fl, sh, ro, bd, hd = build(*seqN[0][:4])
        self.play(Create(cv), FadeIn(fl), FadeIn(sh), FadeIn(ro), FadeIn(bd), FadeIn(hd),
                  *self.cap_anims(seqN[0][4]), run_time=0.9)
        self.wait(1.5)

        def step(tp, fp, n, rate, concl):
            nonlocal cv, fl, sh, ro, bd, hd
            ncv, nfl, nsh, nro, nbd, nhd = build(tp, fp, n, rate)
            self.play(Transform(cv, ncv), Transform(fl, nfl), Transform(sh, nsh),
                      FadeOut(ro), FadeIn(nro), FadeOut(bd), FadeIn(nbd), FadeOut(hd), FadeIn(nhd),
                      *self.cap_anims(concl), run_time=1.1)
            ro, bd, hd = nro, nbd, nhd
            self.wait(1.5)

        # the green area = acceptance probability (the area right of the bar)
        for s in seqN[1:]:
            step(*s)

        self.say("Now hold the sample size at 100 and vary the observed rate.", hold=0.3)
        for s in seqR:
            step(*s)

        self.say("Satisfying, borderline, uncertain, or insufficient — one verdict, set by the data.", hold=2.0)
        self.clear_cap()
        self.play(*[FadeOut(o) for o in self.mobjects], run_time=0.6)


# ════════════════════════════════════════════════════════════════════════════
# SCENE 6 — transfer_test (ROPE): does the metric hold from test to production?
# ════════════════════════════════════════════════════════════════════════════
class TransferTest(Base):
    def construct(self):
        EPS = 0.03
        DLO, DHI = -0.40, 0.70
        rng = np.random.default_rng(3)

        def hdi(s, mass=0.95):
            s = np.sort(s); n = s.size; w = int(np.floor(mass * n))
            wd = s[w:] - s[:n - w]; i = int(np.argmin(wd))
            return float(s[i]), float(s[i + w])

        def decide(delta):
            lo, hi = hdi(delta)
            rho = float(np.mean((delta >= -EPS) & (delta <= EPS)))
            if lo >= -EPS and hi <= EPS:   return "EQUIVALENT", POST_C, rho, lo, hi
            if lo > EPS:                   return "SHIFTED", RED_C, rho, lo, hi
            if hi < -EPS:                  return "SHIFTED", RED_C, rho, lo, hi
            return "UNDECIDED", GRAY_LT, rho, lo, hi

        # ── top panel: the two metric posteriors ────────────────────────
        tax = nice_axis([0.3, 1.0, 0.1], [0, 2.6, 1], 7.6, 1.5).move_to(UP * 1.95)
        txt = VGroup(*[Text(f"{v:.1f}", font_size=15, color=GRAY_LT).next_to(tax.c2p(v, 0), DOWN, buff=0.1)
                       for v in [0.3, 0.5, 0.7, 0.9]])
        TPEAK = 2.1

        def metric_curve(a, b, color):
            xs_ = np.linspace(0.3, 0.999, 400)
            ys = stats.beta(a, b).pdf(xs_); ys = ys / ys.max() * TPEAK
            return curve_from(tax, xs_, ys, color, lw=3)

        def metric_fill(a, b, color):
            xs_ = np.linspace(0.3, 0.999, 400)
            ys = stats.beta(a, b).pdf(xs_); ys = ys / ys.max() * TPEAK
            pts = [tax.c2p(xs_[0], 0), *[tax.c2p(x, y) for x, y in zip(xs_, ys)], tax.c2p(xs_[-1], 0)]
            return Polygon(*pts, color=color, fill_opacity=0.14, stroke_width=0)

        # ── bottom panel: posterior of delta = p_test - p_prod ──────────
        bax = nice_axis([DLO, DHI, 0.1], [0, 2.8, 1], 8.4, 1.7).move_to(DOWN * 2.15)
        bxt = VGroup(*[Text(f"{v:+.1f}".replace("+0.0", "0"), font_size=15, color=GRAY_LT)
                       .next_to(bax.c2p(v, 0), DOWN, buff=0.1) for v in [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6]])
        bxl = MathTex(r"\Delta = p_{\text{test}} - p_{\text{prod}}", font_size=22, color=GRAY_LT).next_to(bax, DOWN, buff=0.32)

        # ROPE band [-eps, +eps] and zero line (persistent)
        rope = Polygon(bax.c2p(-EPS, 0), bax.c2p(EPS, 0), bax.c2p(EPS, 2.7), bax.c2p(-EPS, 2.7),
                       color=CI_C, fill_opacity=0.22, stroke_width=0)
        zline = DashedLine(bax.c2p(0, 0), bax.c2p(0, 2.7), color=GRAY_LT, stroke_width=1.5, dash_length=0.09)
        rope_lbl = Text("ROPE  (±0.03)", font_size=18, color=CI_C).next_to(bax.c2p(0, 2.7), UP, buff=0.08)

        BPEAK = 2.2
        bxs = np.linspace(DLO, DHI, 160)

        def delta_density(delta, color):
            cnt, edges = np.histogram(delta, bins=bxs, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ys = cnt / cnt.max() * BPEAK
            return curve_from(bax, centers, ys, color, lw=3)

        def hdi_bar(lo, hi, color):
            y = 0.32
            bar = Line(bax.c2p(lo, y), bax.c2p(hi, y), color=color, stroke_width=6)
            cap_l = Line(bax.c2p(lo, y - 0.18), bax.c2p(lo, y + 0.18), color=color, stroke_width=4)
            cap_r = Line(bax.c2p(hi, y - 0.18), bax.c2p(hi, y + 0.18), color=color, stroke_width=4)
            return VGroup(bar, cap_l, cap_r)

        def readout(rho, status, color):
            r = MathTex(rf"\rho\ (\text{{ROPE mass}}) = {rho*100:.0f}\%",
                        font_size=28, color=INK).move_to(UP * 0.55)
            b = Text(status, font_size=26, color=color, weight=BOLD)
            box = SurroundingRectangle(b, color=color, buff=0.13, corner_radius=0.1)
            return r, VGroup(box, b).move_to(DOWN * 0.1)

        # scenarios (test counts, prod counts, caption, verdict caption)
        scen = [
            ((180, 20), (60, 40),
             "Test ≈ 0.90, production ≈ 0.60 — both well-sampled.",
             "Shifted: production is meaningfully worse — it does not transfer."),
            ((900, 100), (900, 100),
             "Both sit at ≈ 0.90 on large samples.",
             "Equivalent: the metric holds — it transfers."),
            ((180, 20), (3, 2),
             "Only 5 production images — the difference is barely constrained.",
             "Undecided: the interval straddles the ROPE — gather more data."),
        ]

        self.say("Does the metric measured on the test set survive in production?")
        self.play(Create(tax), FadeIn(txt), run_time=0.6)

        (tt, tp0), (pp, pp0), cap, verd = scen[0]
        at, bt = 1 + tt, 1 + tp0
        ap, bp = 1 + pp, 1 + pp0
        cT, fT = metric_curve(at, bt, POST_C), metric_fill(at, bt, POST_C)
        cP, fP = metric_curve(ap, bp, LIK_C), metric_fill(ap, bp, LIK_C)
        # fixed legend (curves move between scenarios, so don't pin labels to them)
        legend = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.3, color=POST_C, stroke_width=4),
                   Text("test", font_size=20, color=POST_C)).arrange(RIGHT, buff=0.15),
            VGroup(Line(ORIGIN, RIGHT * 0.3, color=LIK_C, stroke_width=4),
                   Text("prod", font_size=20, color=LIK_C)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).to_corner(UL, buff=0.6).shift(DOWN * 0.6)
        self.play(Create(cP), FadeIn(fP), run_time=0.6)
        self.play(Create(cT), FadeIn(fT), FadeIn(legend), run_time=0.6)
        self.say("We sample from each posterior and look at their difference Δ.")

        # build the bottom panel + ROPE
        self.play(Create(bax), FadeIn(bxt), FadeIn(bxl), run_time=0.6)
        self.play(FadeIn(rope), Create(zline), FadeIn(rope_lbl), run_time=0.6)

        def scenario_delta(at, bt, ap, bp):
            dr_t = rng.beta(at, bt, size=60000)
            dr_p = rng.beta(ap, bp, size=60000)
            return dr_t - dr_p

        # ── visualize the Monte-Carlo sampling that builds the Δ posterior ──
        peak_t = stats.beta(at, bt).pdf((at - 1) / (at + bt - 2))
        peak_p = stats.beta(ap, bp).pdf((ap - 1) / (ap + bp - 2))

        def ttick(x, peak, a, b, color):
            h = stats.beta(a, b).pdf(x) / peak * TPEAK
            return Line(tax.c2p(x, 0), tax.c2p(x, h), color=color, stroke_width=3)

        nb = 36
        edges = np.linspace(DLO, DHI, nb + 1)
        counts = np.zeros(nb, dtype=int)
        dots = VGroup()

        def drop(d):
            bi = int(np.clip((d - DLO) / (DHI - DLO) * nb, 0, nb - 1))
            c = counts[bi]; counts[bi] += 1
            cx = 0.5 * (edges[bi] + edges[bi + 1])
            dot = Dot(bax.c2p(cx, 0) + UP * (0.06 + c * 0.085), radius=0.04, color=INK)
            return dot.set_opacity(0.75)

        va = rng.beta(at, bt, size=90)
        vb = rng.beta(ap, bp, size=90)
        vd = va - vb

        self.say("Draw one sample from each posterior, take their difference, drop it on the Δ axis.")
        for i in range(8):          # slow individual draws, with ticks on each curve
            ta = ttick(va[i], peak_t, at, bt, POST_C)
            tb = ttick(vb[i], peak_p, ap, bp, LIK_C)
            dot = drop(vd[i]); dots.add(dot)
            self.play(FadeIn(ta), FadeIn(tb), run_time=0.18)
            self.play(FadeIn(dot, shift=DOWN * 0.15), FadeOut(ta), FadeOut(tb), run_time=0.22)

        self.say("Repeat — thousands of draws build the full posterior of Δ.")
        for batch in (range(8, 30), range(30, 60), range(60, 90)):
            nd = VGroup(*[drop(vd[i]) for i in batch])
            dots.add(*nd)
            self.play(LaggedStart(*[FadeIn(d, shift=DOWN * 0.1) for d in nd], lag_ratio=0.02), run_time=0.5)

        delta = scenario_delta(at, bt, ap, bp)
        status, col, rho, lo, hi = decide(delta)
        dens = delta_density(delta, INK)
        bar = hdi_bar(lo, hi, col)
        ro, bd = readout(rho, status, col)

        # the piled samples become the smooth density (dots fade to a faint rug)
        self.play(Create(dens), dots.animate.set_opacity(0.16), run_time=0.9)
        self.say("Its 95% HDI (bar) versus the ROPE gives the verdict.", hold=0.3)
        self.play(Create(bar), FadeIn(ro), FadeIn(bd), run_time=0.7)
        self.say(verd, hold=1.8)
        self.play(FadeOut(dots), run_time=0.4)

        # remaining scenarios — morph everything in lockstep
        for (tt, tp0), (pp, pp0), cap, verd in scen[1:]:
            at, bt = 1 + tt, 1 + tp0
            ap, bp = 1 + pp, 1 + pp0
            delta = scenario_delta(at, bt, ap, bp)
            status, col, rho, lo, hi = decide(delta)
            nT, nfT = metric_curve(at, bt, POST_C), metric_fill(at, bt, POST_C)
            nP, nfP = metric_curve(ap, bp, LIK_C), metric_fill(ap, bp, LIK_C)
            ndens = delta_density(delta, INK)
            nbar = hdi_bar(lo, hi, col)
            nro, nbd = readout(rho, status, col)
            self.play(Transform(cT, nT), Transform(fT, nfT),
                      Transform(cP, nP), Transform(fP, nfP),
                      Transform(dens, ndens), Transform(bar, nbar),
                      FadeOut(ro), FadeIn(nro), FadeOut(bd), FadeIn(nbd),
                      *self.cap_anims(cap), run_time=1.1)
            ro, bd = nro, nbd
            self.wait(0.8)
            self.play(*self.cap_anims(verd))
            self.wait(1.8)

        self.say("Equivalent, shifted, or undecided — a symmetric test that uses all the uncertainty.", hold=2.0)
        self.clear_cap()
        self.play(*[FadeOut(o) for o in self.mobjects], run_time=0.6)
