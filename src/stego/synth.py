"""
Synthetic cover images.

Used by experiments/make_images.py to fill images/, and by the tests, so
both work from exactly the same pixels.

Why synthetic instead of real photographs?
  * `git clone` and run -- no downloads, no dead links
  * no licensing questions about someone else's photo
  * reproducible: same seed, same pixels, so the numbers in
    experiments/results/ can be checked by anyone

Each image deliberately contains all three kinds of region, because
telling them apart is the whole point of the project:

    smooth gradient  a +-1 change is statistically loud   -> avoid
    fine texture     a +-1 change vanishes into the noise -> use
    hard edges       large local gradients                -> use
"""

import numpy as np

#: The three layouts.  Where the smooth region sits changes how quickly a
#: desynchronizing selector fails, which is why we ship more than one.
LAYOUTS = ("landscape", "inverted", "patchwork")


def make_cover(size=512, seed=0, layout="landscape"):
    """
    Return an (size, size, 3) uint8 RGB image.

    layout
        landscape : smooth sky on top, texture below
        inverted  : texture on top, smooth below
        patchwork : alternating 64px blocks

    A note on noise
    ---------------
    It is tempting to add faint "sensor noise" everywhere for realism.
    Do not.  Even sigma=1.5 randomizes every LSB in the picture, including
    the smooth regions -- and then a perfectly clean image already reads
    ~1.0 LSB entropy and ~0.5 neighbour correlation, so no detector can
    tell clean from stego any more.

    A real photograph behaves like this version: a smooth sky varies
    slowly, so its low bits stay correlated, while textured regions look
    random.  That contrast is the signal this whole project exploits.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}, expected one of {LAYOUTS}")

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)

    # Smooth vertical gradient -- the "sky".  Near-zero local variance, so
    # this is what a good selector learns to avoid.
    sky = 50 + 165 * (yy / size)
    img = np.stack([sky * 0.78, sky * 0.88, sky], axis=-1)

    # Rough texture -- the "gravel".  High local variance, ideal cover.
    if layout == "landscape":
        band = yy > size * 0.55
    elif layout == "inverted":
        band = yy < size * 0.45
    else:
        band = ((yy // 64).astype(int) + (xx // 64).astype(int)) % 2 == 0

    img[band] += rng.normal(0, 34, (size, size, 3))[band]

    # Hard-edged shapes, so the Laplacian feature has something to find.
    for cx, cy, radius, colour in [
        (0.28 * size, 0.30 * size, 0.11 * size, (235, 225, 120)),
        (0.72 * size, 0.22 * size, 0.07 * size, (60, 70, 95)),
    ]:
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 < radius**2
        img[inside] = colour

    # Straight bars: edges with no texture around them.
    img[int(0.44 * size) : int(0.47 * size), :] = (95, 105, 115)
    img[:, int(0.60 * size) : int(0.615 * size)] = (150, 140, 130)

    return np.clip(img, 0, 255).astype(np.uint8)
