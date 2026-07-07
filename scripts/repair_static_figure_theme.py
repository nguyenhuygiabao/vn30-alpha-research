from pathlib import Path
import re

from PIL import Image, ImageOps, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DASHBOARD = REPORTS / "dashboard.html"

DARK_BG = (11, 16, 32)
PANEL_BG = (17, 24, 39)
TEXT_LIGHT = (229, 231, 235)


def is_light_pixel(pixel):
    r, g, b = pixel[:3]
    return r > 210 and g > 210 and b > 210


def is_dark_pixel(pixel):
    r, g, b = pixel[:3]
    return r < 45 and g < 50 and b < 65


def image_light_ratio(img):
    small = img.convert("RGB").resize((120, 80))
    pixels = list(small.getdata())
    return sum(is_light_pixel(p) for p in pixels) / len(pixels)


def image_dark_ratio(img):
    small = img.convert("RGB").resize((120, 80))
    pixels = list(small.getdata())
    return sum(is_dark_pixel(p) for p in pixels) / len(pixels)


def darken_light_chart(path):
    img = Image.open(path).convert("RGB")

    light_ratio = image_light_ratio(img)
    dark_ratio = image_dark_ratio(img)

    # If it is already mostly dark, do not touch it.
    if dark_ratio > 0.55 and light_ratio < 0.25:
        return False, "already dark"

    # Invert light matplotlib-style charts into dark charts.
    inverted = ImageOps.invert(img)

    # Shift near-black inverted background to navy.
    px = inverted.load()
    w, h = inverted.size

    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]

            # Background / plot area after inversion
            if r < 45 and g < 45 and b < 55:
                px[x, y] = DARK_BG

            # Slightly lighter dark grid/axes areas
            elif r < 80 and g < 85 and b < 100:
                px[x, y] = PANEL_BG

            # Avoid neon whites after inversion
            elif r > 235 and g > 235 and b > 235:
                px[x, y] = TEXT_LIGHT

    # Improve readability slightly.
    inverted = ImageEnhance.Contrast(inverted).enhance(1.08)
    inverted = ImageEnhance.Sharpness(inverted).enhance(1.12)

    path.write_bytes(b"")
    inverted.save(path, "PNG", optimize=True)

    return True, f"darkened light_ratio={light_ratio:.2f}, dark_ratio={dark_ratio:.2f}"


def patch_dashboard_static_css():
    if not DASHBOARD.exists():
        return False

    text = DASHBOARD.read_text(encoding="utf-8", errors="replace")
    original = text

    patch = """
<style>
  .figure-card,
  .static-card,
  .diagnostic-card {
    background: #0b1020 !important;
  }

  .figure-card img,
  .static-card img,
  .diagnostic-card img,
  img[src*="figures/"] {
    background: #0b1020 !important;
    object-fit: contain !important;
  }

  .figure-card .figure-frame,
  .static-card .figure-frame,
  .diagnostic-card .figure-frame,
  div:has(> img[src*="figures/"]) {
    background: #0b1020 !important;
  }
</style>
"""

    marker_start = "<!-- VN30_STATIC_FIGURE_POLISH_START -->"
    marker_end = "<!-- VN30_STATIC_FIGURE_POLISH_END -->"
    full_patch = marker_start + patch + marker_end

    while marker_start in text and marker_end in text:
        before, rest = text.split(marker_start, 1)
        _, after = rest.split(marker_end, 1)
        text = before + after

    if "</head>" in text:
        text = text.replace("</head>", full_patch + "\n</head>", 1)
    else:
        text = full_patch + "\n" + text

    if text != original:
        DASHBOARD.write_text(text, encoding="utf-8", newline="\n")
        return True

    return False


def main():
    if not FIGURES.exists():
        raise SystemExit(f"Missing figures directory: {FIGURES}")

    changed = []

    for path in sorted(FIGURES.glob("*.png")):
        touched, reason = darken_light_chart(path)
        print(f"{path.name}: {reason}")
        if touched:
            changed.append(path.name)

    css_changed = patch_dashboard_static_css()

    print()
    print(f"Static PNGs darkened: {len(changed)}")
    for name in changed:
        print(f"  {name}")

    print(f"Dashboard static CSS patched: {css_changed}")


if __name__ == "__main__":
    main()
