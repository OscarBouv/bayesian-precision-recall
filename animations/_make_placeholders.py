"""Generate clean placeholder photo cards for cat / dog until real photos are dropped in."""
from PIL import Image, ImageDraw
from pathlib import Path
import math

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)

S = 512  # square canvas


def rounded_card(bg_top, bg_bot):
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # vertical gradient background
    for y in range(S):
        t = y / S
        r = int(bg_top[0] * (1 - t) + bg_bot[0] * t)
        g = int(bg_top[1] * (1 - t) + bg_bot[1] * t)
        b = int(bg_top[2] * (1 - t) + bg_bot[2] * t)
        d.line([(0, y), (S, y)], fill=(r, g, b, 255))
    # rounded mask
    mask = Image.new("L", (S, S), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, S - 1, S - 1], radius=48, fill=255)
    img.putalpha(mask)
    return img, ImageDraw.Draw(img)


def draw_cat(d, cx, cy, r, col):
    # ears
    d.polygon([(cx - r*0.7, cy - r*0.5), (cx - r*0.95, cy - r*1.15), (cx - r*0.25, cy - r*0.75)], fill=col)
    d.polygon([(cx + r*0.7, cy - r*0.5), (cx + r*0.95, cy - r*1.15), (cx + r*0.25, cy - r*0.75)], fill=col)
    # head
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=10)
    # eyes
    d.ellipse([cx - r*0.45, cy - r*0.2, cx - r*0.2, cy + r*0.1], fill=col)
    d.ellipse([cx + r*0.2, cy - r*0.2, cx + r*0.45, cy + r*0.1], fill=col)
    # nose
    d.polygon([(cx - r*0.1, cy + r*0.25), (cx + r*0.1, cy + r*0.25), (cx, cy + r*0.42)], fill=(245, 197, 66))
    # whiskers
    for dy in (0.30, 0.42):
        d.line([(cx - r*0.15, cy + r*dy), (cx - r*0.85, cy + r*(dy - 0.08))], fill=col, width=5)
        d.line([(cx + r*0.15, cy + r*dy), (cx + r*0.85, cy + r*(dy - 0.08))], fill=col, width=5)


def draw_dog(d, cx, cy, r, col):
    # floppy ears
    d.ellipse([cx - r*1.25, cy - r*0.8, cx - r*0.55, cy + r*0.55], outline=col, width=10)
    d.ellipse([cx + r*0.55, cy - r*0.8, cx + r*1.25, cy + r*0.55], outline=col, width=10)
    # head
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=10)
    # eyes
    d.ellipse([cx - r*0.45, cy - r*0.25, cx - r*0.22, cy], fill=col)
    d.ellipse([cx + r*0.22, cy - r*0.25, cx + r*0.45, cy], fill=col)
    # snout
    d.ellipse([cx - r*0.5, cy + r*0.15, cx + r*0.5, cy + r*0.8], outline=col, width=8)
    d.ellipse([cx - r*0.13, cy + r*0.45, cx + r*0.13, cy + r*0.62], fill=(245, 197, 66))


# cat — cool blue card
img, d = rounded_card((58, 92, 140), (33, 54, 84))
draw_cat(d, S/2, S/2 - 10, 150, (236, 236, 236))
img.save(ASSETS / "cat.png")

# dog — warm card
img, d = rounded_card((150, 96, 58), (90, 56, 33))
draw_dog(d, S/2, S/2 - 10, 140, (236, 236, 236))
img.save(ASSETS / "dog.png")

print("wrote", ASSETS / "cat.png", "and dog.png")
