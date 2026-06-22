"""
Animation 1: Bayesian posterior updating
"""

from manim import *
import numpy as np
from scipy import stats

config.tex_template = TexTemplate()  # fallback — won't be used


class PosteriorUpdate(Scene):
    def construct(self):
        batches = [
            (1, 0), (0, 1), (1, 0), (1, 0),
            (0, 1), (1, 0), (1, 0), (1, 0),
            (0, 1), (1, 0), (1, 0), (0, 1),
            (1, 0), (1, 0), (1, 0), (1, 0),
            (0, 1), (1, 0), (1, 0), (1, 0),
        ]

        alpha, beta_ = 1.0, 1.0
        tp, fp = 0, 0

        ax = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 9, 2],
            x_length=9,
            y_length=5,
            axis_config={"color": WHITE, "stroke_width": 2, "include_numbers": False},
        ).to_edge(DOWN, buff=1.0)

        # Manual x tick labels (plain Text, no LaTeX)
        for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            lbl = Text(f"{v:.1f}", font_size=18, color=GRAY).next_to(ax.c2p(v, 0), DOWN, buff=0.15)
            self.add(lbl)

        x_lbl = Text("Precision", font_size=22, color=GRAY).next_to(ax, DOWN, buff=0.55)
        y_lbl = Text("Density", font_size=22, color=GRAY).next_to(ax, LEFT, buff=0.3).rotate(PI / 2)
        title = Text("Bayesian Posterior Updating", font_size=34, color=WHITE).to_edge(UP)
        self.add(ax, x_lbl, y_lbl, title)

        xs = np.linspace(0.001, 0.999, 400)

        def make_curve(a, b, color):
            ys = stats.beta(a, b).pdf(xs)
            pts = [ax.c2p(x, y) for x, y in zip(xs, ys)]
            return VMobject(color=color, stroke_width=3).set_points_smoothly(pts)

        def mode_line(a, b):
            m = (a - 1) / (a + b - 2) if (a + b) > 2 else 0.5
            return DashedLine(ax.c2p(m, 0), ax.c2p(m, 7.5), color=YELLOW, stroke_width=1.5, dash_length=0.12)

        counter = Text("Prior: Beta(1,1)  —  uniform", font_size=22, color=YELLOW).next_to(title, DOWN, buff=0.18)
        self.add(counter)

        curve = make_curve(alpha, beta_, BLUE_B)
        self.play(Create(curve), run_time=0.8)
        self.wait(0.4)

        m_line = None

        for dtp, dfp in batches:
            tp += dtp
            fp += dfp
            a, b = alpha + tp, beta_ + fp
            t = (tp + fp) / len(batches)

            new_counter = Text(
                f"TP={tp}  FP={fp}  observed={tp/(tp+fp):.0%}",
                font_size=22, color=YELLOW,
            ).next_to(title, DOWN, buff=0.18)

            new_curve = make_curve(a, b, interpolate_color(BLUE, TEAL, t))
            new_ml = mode_line(a, b)
            anims = [Transform(curve, new_curve), Transform(counter, new_counter)]
            if m_line is None:
                m_line = new_ml
                anims.append(FadeIn(m_line))
            else:
                anims.append(Transform(m_line, new_ml))
            self.play(*anims, run_time=0.32)

        a, b = alpha + tp, beta_ + fp
        mv = (a - 1) / (a + b - 2)
        end_lbl = Text(f"mode = TP/(TP+FP) = {mv:.2f}", font_size=24, color=YELLOW)
        end_lbl.next_to(ax.c2p(mv, 6.5), RIGHT, buff=0.15)
        self.play(Write(end_lbl))
        self.wait(1.5)
