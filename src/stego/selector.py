"""
Which pixels do we hide in, and in what order?

Every selector answers that one question: given an image, return an array
of flat pixel indices.  Same interface for all of them, so
experiments/exp2_compare.py can benchmark them in a single loop.

THE ONE RULE EVERY SELECTOR MUST OBEY
-------------------------------------
    select(cover) must equal select(stego)

The receiver only has the stego image.  If it computes a different list
than the sender did, it reads the bits out of the wrong pixels.  And
because reading is positional -- 1st bit, 2nd bit, 3rd bit -- one extra
or missing index shifts every bit after it and destroys the rest of the
message.

Sequential and Random obey the rule trivially: they never look at pixel
values at all.  Variance and ML obey it because they build their features
from features.stabilize(), which erases the bit we modify.

NaiveMLSelector deliberately breaks the rule.  It exists so that
exp1_desync.py can measure the failure and test_sync.py can lock it down.
"""

import numpy as np

from .features import extract

#: Fixed seed so RandomSelector gives the same answer to sender and
#: receiver.  In a real system this would be derived from a shared
#: passphrase; here it is a constant because there is no key exchange.
RANDOM_SEED = 12345

#: Threshold for VarianceSelector.  Roughly: "is this neighbourhood
#: textured enough that a +-1 change disappears into it?"
VARIANCE_THRESHOLD = 20.0


class Selector:
    """Base class.  Subclasses implement select()."""

    name = "base"

    def select(self, img):
        """(H, W, 3) image -> 1-D array of flat pixel indices to use."""
        raise NotImplementedError

    def capacity_bits(self, img):
        """How many bits this selector can hide in this image."""
        return len(self.select(img))

    def __repr__(self):
        return f"<{type(self).__name__} name={self.name!r}>"


def _usable_mask(shape):
    """
    True for every pixel except a 1-pixel border.

    The 3x3 filters in features.py have to invent values at the edges,
    and those invented values are not reliable, so we skip the border.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[1:-1, 1:-1] = True
    return mask.ravel()


class SequentialSelector(Selector):
    """
    Baseline: top-left to bottom-right, in order.

    The classic textbook LSB method.  Easy to detect, because an analyst
    sees the first N% of the image looking random and the rest looking
    normal -- a very obvious signature.
    """

    name = "sequential"

    def select(self, img):
        return np.flatnonzero(_usable_mask(img.shape))


class RandomSelector(Selector):
    """
    Baseline: same pixels, shuffled order.

    Spreads changes over the whole image instead of concentrating them at
    the top, which removes the "front section is different" signature.
    But it still ignores content -- it will happily write into a flat sky.
    """

    name = "random"

    def __init__(self, seed=RANDOM_SEED):
        self.seed = seed

    def select(self, img):
        idx = np.flatnonzero(_usable_mask(img.shape))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(idx)
        return idx


class VarianceSelector(Selector):
    """
    Baseline: one hand-written rule -- keep pixels whose 3x3 variance is
    above a threshold.

    This is the honest comparison for the ML model.  If the Random Forest
    cannot beat a single `if`, the ML adds nothing, and we should say so.
    """

    name = "variance"

    def __init__(self, threshold=VARIANCE_THRESHOLD):
        self.threshold = threshold

    def select(self, img):
        feats = extract(img, stable=True)          # stable -> sync-safe
        good = feats[:, 1] > self.threshold        # column 1 is variance
        return np.flatnonzero(good & _usable_mask(img.shape))


class MLSelector(Selector):
    """
    The real thing: a trained Random Forest decides pixel by pixel.

    The model sees [brightness, variance, edge] and outputs 1 (safe to
    embed) or 0 (leave alone).  Because the features come from the
    stabilized image, sender and receiver always agree.
    """

    name = "ml"

    def __init__(self, model):
        self.model = model

    def select(self, img):
        feats = extract(img, stable=True)          # <- the fix
        good = self.model.predict(feats) == 1
        return np.flatnonzero(good & _usable_mask(img.shape))


class NaiveMLSelector(MLSelector):
    """
    The BROKEN version, kept on purpose.

    Identical to MLSelector except `stable=False`: it reads features from
    the raw image, including the very bit we overwrite.  So embedding
    changes the features, which changes the model's answer, which changes
    the pixel list -- and the receiver desyncs.

    Do not use this to hide real messages.  It exists as the "before"
    half of the experiment.
    """

    name = "ml-naive"

    def select(self, img):
        feats = extract(img, stable=False)         # <- the bug
        good = self.model.predict(feats) == 1
        return np.flatnonzero(good & _usable_mask(img.shape))
