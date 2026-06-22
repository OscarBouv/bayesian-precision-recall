"""
Animation 2: compare_models — P(model_a > model_b)
"""

from manim import *
import numpy as np
from scipy import stats


class CompareModels(Scene):
    def construct(self):
        a_tp, a_fp = 80, 20
        b_tp, b_fp = 72, 26
        alpha, beta_ = 1.0, 1.0

        dist_a = stats.beta(alpha + a_tp, beta_ + a_fp)
        dist_b = stats.beta(alpha + b_tp, beta_ + b_fp)

        rng = np.random.default_rng(42)
        prob = float((dist_a.rvs(200_000, random_state=rng) > dist_b.rvs(200_000, random_state=rng)).mean())

        ax = Axes(
            x_range=[0.5, 1.0, 0.1],
            y_range=[0, 14, 2],
            x_length=9,
            y_length=5,
            axis_config={"color": WHITE, "stroke_width": 2, "include_numbers": False},
        ).to_edge(DOWN, buff=1.0)

        for v in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            self.add(Text(f"{v:.1f}", font_size=18, color=GRAY).next_to(ax.c2p(v, 0), DOWN, buff=0.15))

        x_lbl = Text("Precision", font_size=22, color=GRAY).next_to(ax, DOWN, buff=0.55)
        title  = Text("Comparing Model Posteriors", font_size=34, color=WHITE).to_edge(UP)
        self.add(ax, x_lbl, title)

        xs = np.linspace(0.50, 1.0, 500)

        def curve(dist, color):
            pts = [ax.c2p(x, dist.pdf(x)) for x in xs]
            return VMobject(color=color, stroke_width=3).set_points_smoothly(pts)

        def fill_poly(dist, color):
            pts = ([ax.c2p(xs[0], 0)]
                   + [ax.c2p(x, dist.pdf(x)) for x in xs]
                   + [ax.c2p(xs[-1], 0)])
            return Polygon(*pts, color=color, fill_opacity=0.18, stroke_width=0)

        cb = curve(dist_b, ORANGE)
        fb = fill_poly(dist_b, ORANGE)
        lb = Text("Model B", font_size=22, color=ORANGE).move_to(ax.c2p(0.725, 11.5))

        ca = curve(dist_a, BLUE)
        fa = fill_poly(dist_a, BLUE)
        la = Text("Model A", font_size=22, color=BLUE).move_to(ax.c2p(0.815, 11.5))

        self.play(Create(cb), FadeIn(fb), Write(lb), run_time=0.8)
        self.play(Create(ca), FadeIn(fa), Write(la), run_time=0.8)
        self.wait(0.4)

        diff_ys = np.maximum(dist_a.pdf(xs) - dist_b.pdf(xs), 0)
        diff_pts = ([ax.c2p(xs[0], 0)]
                    + [ax.c2p(x, y) for x, y in zip(xs, diff_ys)]
                    + [ax.c2p(xs[-1], 0)])
        diff_poly = Polygon(*diff_pts, color=GREEN, fill_opacity=0.42, stroke_width=0)

        prob_text = Text(f"P(A > B) = {prob:.0%}", font_size=30, color=GREEN).to_corner(UR, buff=0.9)
        sub_text  = Text("accounting for uncertainty in both estimates", font_size=19, color=GRAY).next_to(prob_text, DOWN, buff=0.12)

        self.play(FadeIn(diff_poly), Write(prob_text), Write(sub_text), run_time=1.0)
        self.wait(2.0)
