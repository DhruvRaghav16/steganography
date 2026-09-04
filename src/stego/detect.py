"""
Steganalysis: look at an image and guess whether it carries hidden data.

This is the attacker's half of the project.  Building it is what forces
you to think about what your own embedder leaves behind.

Three measurements, cheapest first:

    lsb_entropy       are the last bits balanced 50/50?
    lsb_correlation   do neighbouring last bits still resemble each other?
    chi_square        did the value pairs (0,1), (2,3), ... even out?

WHICH ONE ACTUALLY WORKS -- READ THIS
-------------------------------------
Only lsb_correlation reliably separates clean from stego on the images in
this repo.  Measured on images/cover1_landscape.png with a random payload:

    fill      correlation    entropy
    clean        0.776        1.000
     10%         0.721        1.000
     25%         0.641        1.000
     50%         0.506        1.000
    100%         0.453        1.000

Entropy never moves, because entropy only measures the *balance* of ones
and zeros -- and a clean photo's LSB plane is already about half ones.
It is blind to spatial structure, so a stripey plane and a random plane
score exactly the same.  This is precisely the mistake the original
version of this project made: it flagged "entropy > 0.95" as evidence of
hiding, which fires on ordinary photographs.

Correlation is the one that works, because it asks a structural question:
in a real image, neighbouring pixels have similar values, so their low
bits agree far more often than chance.  Overwriting them with message
bits pushes that agreement towards 0.5 -- a coin flip.

chi_square is included because it is the textbook attack and worth
understanding, but it reads ~0 on every image here, clean or not: these
synthetic covers use too few distinct blue values for the pair statistics
to mean anything.  It is kept, and kept honest, rather than quietly
dropped -- see its docstring.
"""

import numpy as np
from scipy.stats import chi2

from .features import BLUE

#: Clean natural images sit well above this; heavily-embedded ones fall
#: towards 0.5.  Calibrate on your own corpus before trusting it -- the
#: right value depends on how smooth your images are.
CORRELATION_SUSPICIOUS = 0.60


def lsb_entropy(img, channel=BLUE):
    """
    Shannon entropy of the LSB plane, in bits.  Range 0.0 to 1.0.

    WARNING: near-useless as a detector, and included to show why.  It
    only counts ones vs zeros, so it cannot tell an ordered plane from a
    random one.  A clean photo typically reads ~1.0 already.
    """
    bits = (img[:, :, channel] & 1).ravel()
    p = float(bits.mean())
    if p in (0.0, 1.0):
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def lsb_correlation(img, channel=BLUE):
    """
    Fraction of horizontally-adjacent LSB pairs that are equal.  The
    detector that actually works here.

    0.5  = neighbours agree exactly as often as chance -> looks embedded
    0.75 = strong local structure                      -> looks natural

    Natural images are locally smooth, so neighbouring pixels usually
    share their low bit.  Message bits destroy that.
    """
    bits = (img[:, :, channel] & 1).astype(np.int8)
    horizontal = (bits[:, :-1] == bits[:, 1:]).mean()
    vertical = (bits[:-1, :] == bits[1:, :]).mean()
    return float((horizontal + vertical) / 2)


def chi_square(img, channel=BLUE):
    """
    The classic pairs-of-values attack.  Returns a p-value in [0, 1],
    where HIGH means suspicious.

    The idea: LSB embedding can only move a value between the two members
    of a pair -- 200 <-> 201, 202 <-> 203 -- never from 201 to 202.
    Writing random bits therefore drives the two counts in each pair
    towards equality.  In a natural image they differ noticeably.

    CAVEAT: on the synthetic covers in images/ this reads ~0 for both
    clean and stego images.  Those covers use relatively few distinct blue
    values, so most pairs are near-empty and the statistic is dominated by
    a handful of very unequal ones.  The test is sound; the images do not
    suit it.  On a real photograph with a full-range histogram it becomes
    informative.  Reported as-is rather than tuned until it looks good.
    """
    hist = np.bincount(img[:, :, channel].ravel(), minlength=256).astype(float)
    even, odd = hist[0::2], hist[1::2]

    expected = (even + odd) / 2.0
    keep = expected > 4                    # chi-square needs decent counts
    if keep.sum() < 2:
        return 0.0

    observed = even[keep]
    expected = expected[keep]
    statistic = float(((observed - expected) ** 2 / expected).sum())

    # Survival function = P(chi2 >= statistic).  A small statistic (counts
    # already equal) gives a p-value near 1, i.e. suspicious.
    return float(chi2.sf(statistic, int(keep.sum()) - 1))


def analyze(img):
    """Run all three and return the numbers plus a cautious verdict."""
    correlation = lsb_correlation(img)
    entropy = lsb_entropy(img)
    p_value = chi_square(img)

    if correlation < 0.52:
        verdict = "strong signal: LSB plane is essentially random"
    elif correlation < CORRELATION_SUSPICIOUS:
        verdict = "possible embedding: neighbouring LSBs barely correlate"
    else:
        verdict = "nothing unusual: LSB plane retains natural structure"

    return {
        "lsb_correlation": correlation,
        "lsb_entropy": entropy,
        "chi_square_p": p_value,
        "verdict": verdict,
    }


def psnr(a, b):
    """
    Peak signal-to-noise ratio in dB -- how close two images are.

    Higher is better.  Above ~40 dB the difference is invisible to the
    eye; LSB embedding typically lands above 60, which is exactly why
    nobody notices it.  Note that PSNR measures VISIBILITY, not
    detectability -- an image can be statistically obvious and still
    score 64 dB.  Returns infinity when the images are identical.
    """
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((255.0**2) / mse))
