"""
Animation 3: transfer_test — distributional consistency
"""

from manim import *
import numpy as np
from scipy import stats


class TransferTest(Scene):
    def construct(self):
        alpha, beta_ = 1.0, 1.0
        ref_tp, ref_fp = 90, 22
        evl_tp, evl_fp = 15, 25

        dist_ref = stats.beta(alpha + ref_tp, beta_ + ref_fp)
        dist_evl = stats.beta(alpha + evl_tp, beta_ + evl_fp)

        mu_ref = dist_ref.mean()
        S = min(dist_evl.cdf(mu_ref), 1 - dist_evl.cdf(mu_ref))

        ax = Axes(
            x_range=[0.0, 1.0, 0.2],
            y_range=[0, 14, 2],
            x_length=9,
            y_length=5,
            axis_config={"color": WHITE, "stroke_width": 2, "include_numbers": False},
        ).to_edge(DOWN, buff=1.0)

        for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            self.add(Text(f"{v:.1f}", font_size=18, color=GRAY).next_to(ax.c2p(v, 0), DOWN, buff=0.15))

        x_lbl = Text("Precision", font_size=22, color=GRAY).next_to(ax, DOWN, buff=0.55)
        title  = Text("Distributional Consistency Test", font_size=34, color=WHITE).to_edge(UP)
        self.add(ax, x_lbl, title)

        xs = np.linspace(0.001, 0.999, 500)

        def curve(dist, color):
            pts = [ax.c2p(x, dist.pdf(x)) for x in xs]
            return VMobject(color=color, stroke_width=3).set_points_smoothly(pts)

        # Reference
        cr = curve(dist_ref, BLUE)
        lr = Text("dist_A  (reference)", font_size=22, color=BLUE).to_corner(UL, buff=1.1)
        self.play(Create(cr), Write(lr), run_time=0.8)

        # mu_ref line
        mu_line = DashedLine(ax.c2p(mu_ref, 0), ax.c2p(mu_ref, 12.5),
                             color=YELLOW, stroke_width=2, dash_length=0.15)
        mu_lbl  = Text(f"mu_ref = {mu_ref:.2f}", font_size=23, color=YELLOW)
        mu_lbl.next_to(ax.c2p(mu_ref, 12.5), UP, buff=0.08)
        self.play(Create(mu_line), Write(mu_lbl), run_time=0.6)
        self.wait(0.5)

        # Eval
        ce = curve(dist_evl, ORANGE)
        le = Text("dist_B  (evaluation)", font_size=22, color=ORANGE).next_to(lr, DOWN, buff=0.2, aligned_edge=LEFT)
        self.play(Create(ce), Write(le), run_time=1.0)
        self.wait(0.4)

        # Tail shading — left tail of eval up to mu_ref = S
        tail_xs = xs[xs <= mu_ref]
        tail_pts = ([ax.c2p(tail_xs[0], 0)]
                    + [ax.c2p(x, dist_evl.pdf(x)) for x in tail_xs]
                    + [ax.c2p(tail_xs[-1], 0)])
        tail = Polygon(*tail_pts, color=RED, fill_opacity=0.52, stroke_width=0)

        s_text = Text(f"S = {S:.4f}  <=  0.05", font_size=26, color=RED).to_corner(UR, buff=0.9)
        verdict = Text("posteriors inconsistent", font_size=21, color=RED).next_to(s_text, DOWN, buff=0.12)
        formula = Text("S = min( F_eval(mu_ref),  1 - F_eval(mu_ref) )", font_size=18, color=GRAY)
        formula.next_to(verdict, DOWN, buff=0.15)

        self.play(FadeIn(tail), Write(s_text), Write(verdict), Write(formula), run_time=1.0)
        self.wait(2.0)
