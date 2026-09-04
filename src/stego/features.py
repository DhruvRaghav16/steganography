"""
Turn every pixel into 3 numbers that describe its neighbourhood.

THE KEY IDEA OF THIS WHOLE PROJECT
----------------------------------
We hide data in the last bit (the LSB) of the BLUE channel.

So before we measure anything, we set every blue LSB to 0.

That makes the features *blind* to the exact bit we are about to change.
The sender looks at the cover image, the receiver looks at the stego
image -- but after stabilize() those two images are bit-for-bit identical,
so both sides compute the same features and pick the same pixels.

Without this step the two sides disagree about where the data lives, and
because reading is positional a single disagreement shifts everything
after it.  experiments/exp1_desync.py measures the damage;
tests/test_sync.py makes sure it can never come back.
"""

import numpy as np
from scipy.ndimage import laplace, uniform_filter

#: Size of the neighbourhood we look at (3x3 window around each pixel).
WINDOW = 3

#: Human-readable names, in the same order as the columns of extract().
FEATURE_NAMES = ("brightness", "variance", "edge")

#: Index of the channel we hide data in.  0=R, 1=G, 2=B.
#: Blue is chosen because the eye is least sensitive to it (see to_gray).
BLUE = 2


def stabilize(img):
    """
    Zero out the blue LSB.

    `0xFE` is `11111110` in binary, so `value & 0xFE` clears the last bit
    and leaves the other seven alone.  A cover image and its stego image
    differ *only* in those last bits, so after this call they are equal.
    """
    out = img.copy()
    out[:, :, BLUE] &= 0xFE
    return out


def to_gray(img):
    """
    Convert RGB to a single brightness value (ITU-R BT.601 luma).

    Note the weights: green counts for 58.7% of perceived brightness but
    blue for only 11.4%.  That is exactly why we hide in blue -- a change
    there disturbs the picture the least.
    """
    return (
        0.299 * img[:, :, 0]
        + 0.587 * img[:, :, 1]
        + 0.114 * img[:, :, 2]
    ).astype(np.float64)


def extract(img, stable=True):
    """
    (H, W, 3) uint8 image  ->  (H*W, 3) float feature matrix.

    Row `i` of the result describes flat pixel `i`, so it lines up with
    `img.reshape(-1, 3)` -- that is the indexing every Selector uses.

    Parameters
    ----------
    stable : bool
        True  -> strip the blue LSB first (correct, sync-safe).
        False -> use the raw image (the original broken behaviour,
                 kept so NaiveMLSelector can demonstrate the bug).

    The three features
    ------------------
    brightness : how light or dark the pixel is (0-255).
    variance   : how much the 3x3 neighbourhood varies.  This is the
                 texture measure and by far the most useful of the three
                 -- flat sky scores ~0, gravel scores in the hundreds.
    edge       : |Laplacian|, the second derivative.  Zero on flat areas
                 AND on smooth gradients, large at real boundaries.  That
                 is what makes it different from variance.
    """
    gray = to_gray(stabilize(img) if stable else img)

    # Var(X) = E[X^2] - E[X]^2, and each E[] is just a box blur.
    # Doing it this way computes the variance of every 3x3 window in the
    # whole image at once -- no Python loop over a million pixels.
    mean = uniform_filter(gray, WINDOW)
    mean_of_squares = uniform_filter(gray * gray, WINDOW)
    variance = np.maximum(mean_of_squares - mean * mean, 0.0)  # clamp fp noise

    edge = np.abs(laplace(gray))

    return np.stack([gray.ravel(), variance.ravel(), edge.ravel()], axis=1)
