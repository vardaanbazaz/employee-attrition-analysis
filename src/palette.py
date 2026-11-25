"""
Matte earth palette shared by every figure written to assets/.

Matches the portfolio site (wabi-sabi style). Imported by
scripts/generate_visuals.py and src/evaluate.py; apply_theme() sets the
matplotlib rcParams both scripts rely on.

Series colours are spaced in CIELAB lightness (L* ~24 / 32 / 39 / 46 / 53)
so they stay distinguishable in greyscale; every series and reference colour
is at least 3:1 against BACKGROUND, and both text colours at least 4.5:1.
Grid and the light colormap ramp stops are non-data and exempt from the 3:1
rule. The SHAP mid stop is the exception: mid-valued dots are data, so it
reaches 3:1 too.
Numbers printed inside heatmap / confusion-matrix cells use CELL_TEXT or
BACKGROUND, picked per cell by text_color_for, and reach 4.5:1 on every cell.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# Surfaces and ink
BACKGROUND = "#F1F0EB"   # linen
TEXT = "#2B2A27"         # charcoal
CELL_TEXT = "#22221F"    # TEXT darkened minimally so mid-rust heatmap cells reach 4.5:1
MUTED_TEXT = "#6B6A64"
GRID = "#D5D2C9"         # soft ash
REFERENCE = "#7A786F"    # no-skill / baseline lines, spines

# Series, darkest to lightest
CHARCOAL = "#3A3935"     # headline model
DEEP_TEAL = "#265252"    # stayed / retained
MOSS = "#546131"
RUST = "#A4573B"         # left / attrition
OCHRE = "#9C7A26"

LEFT_COLOR = RUST
STAYED_COLOR = DEEP_TEAL

HEADLINE_LINEWIDTH = 2.8
SERIES_LINEWIDTH = 1.8
REFERENCE_LINESTYLE = (0, (5, 3))  # dashed

# Colormap teal pole, as specified (no "stayed" meaning in the colormaps)
TEAL_STOP = "#3F6B6B"

# Diverging (correlation heatmap): teal -> linen -> rust
DIVERGING_CMAP = LinearSegmentedColormap.from_list("earth_diverging", [TEAL_STOP, BACKGROUND, RUST])
# Sequential (confusion matrix): linen -> rust
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("earth_sequential", [BACKGROUND, RUST])
# SHAP feature value: low -> high. Mid stop is a grey at 3.11:1 on BACKGROUND so mid-valued dots stay visible.
SHAP_MID = "#8A8880"
SHAP_CMAP = LinearSegmentedColormap.from_list("earth_shap", [TEAL_STOP, SHAP_MID, RUST])


def _relative_luminance(color) -> float:
    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(c) for c in to_rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b) -> float:
    la, lb = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def text_color_for(fill) -> str:
    """Returns CELL_TEXT or BACKGROUND, whichever contrasts more with the given cell fill."""
    return CELL_TEXT if _contrast(CELL_TEXT, fill) >= _contrast(BACKGROUND, fill) else BACKGROUND


def apply_theme() -> None:
    """Sets matplotlib rcParams for the matte earth look. Fonts are left unchanged."""
    plt.rcParams.update({
        'figure.facecolor': BACKGROUND,
        'axes.facecolor': BACKGROUND,
        'savefig.facecolor': BACKGROUND,
        'savefig.edgecolor': 'none',
        'text.color': TEXT,
        'axes.labelcolor': TEXT,
        'axes.titlecolor': TEXT,
        'xtick.color': MUTED_TEXT,
        'ytick.color': MUTED_TEXT,
        'xtick.labelcolor': MUTED_TEXT,
        'ytick.labelcolor': MUTED_TEXT,
        'axes.edgecolor': REFERENCE,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': GRID,
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'grid.alpha': 1.0,
        'legend.facecolor': BACKGROUND,
        'legend.edgecolor': GRID,
        'legend.labelcolor': TEXT,
    })
