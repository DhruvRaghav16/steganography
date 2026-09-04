"""
THE most important test in the repo.

It pins down the invariant that makes this whole design work:

    select(cover) == select(stego)

If embedding ever changes the pixel list -- even by one index -- the
receiver reads from the wrong pixels, and because reading is positional,
everything after that point is garbage.  That is exactly the bug the
original version had (96% payload loss at scale, see
experiments/exp1_desync.py).

First test: lock in the fix (stable features).
Second test: the broken variant must STILL fail.  Keeping a failing test
around is a promise: "this bug is known, understood, and must never be
re-introduced."  It also proves the first test is not vacuous -- there
really is something to prevent.
"""

import numpy as np
import pytest

from stego.core import hide
from stego.selector import (
    MLSelector,
    NaiveMLSelector,
    RandomSelector,
    SequentialSelector,
    VarianceSelector,
)
from stego.synth import LAYOUTS, make_cover
from stego.training import train

#: Small enough to keep the suite fast, big enough that the classifier
#: has a real decision to make.
SIZE = 128


@pytest.fixture(scope="module")
def model():
    """
    A small forest trained on the same synthetic covers the experiments
    use.  Module-scoped so it is fitted once for the whole file.
    """
    covers = [make_cover(SIZE, seed=i, layout=l) for i, l in enumerate(LAYOUTS)]
    trained, _stats = train(covers, n_estimators=12, seed=0, verbose=False)
    return trained


@pytest.fixture
def cover():
    return make_cover(SIZE, seed=99, layout="landscape")


def test_stable_selectors_are_invariant(model, cover):
    """The four safe selectors must give identical lists on both sides."""
    for selector in [
        SequentialSelector(),
        RandomSelector(),
        VarianceSelector(),
        MLSelector(model),
    ]:
        stego = hide(cover, "A" * 200, selector)
        np.testing.assert_array_equal(
            selector.select(cover),
            selector.select(stego),
            err_msg=f"{selector.name}: selection changed after embedding",
        )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_naive_selector_desyncs(model, layout):
    """
    The original bug is still there -- and stays there.

    NaiveMLSelector reads features from the raw image, including the LSB
    we overwrite.  Embedding therefore changes its answer.  If this test
    ever starts PASSING on its own, something subtle changed; the naive
    selector is supposed to be wrong.

    Two details that make this test honest:

    * the message fills half the available pixels, so the embedded bits
      actually reach the unstable ones sitting on the classifier's
      decision boundary.  A short message might only touch rock-stable
      pixels and appear to work -- which is exactly how the original
      project fooled itself with a 45-character demo.
    * all three layouts are checked, because where the smooth region sits
      changes how quickly the failure shows up.
    """
    cover = make_cover(SIZE, seed=99, layout=layout)
    selector = NaiveMLSelector(model)

    n_pixels = selector.select(cover).size
    assert n_pixels > 0, "test image is too boring -- no pixels selected"

    stego = hide(cover, "A" * (n_pixels // 16), selector)

    assert not np.array_equal(
        selector.select(cover), selector.select(stego)
    ), f"expected the naive selector to desync on {layout}, but it did not"


def test_ml_selector_uses_a_reasonable_fraction(model, cover):
    """The model should agree with common sense: some pixels are usable,
    not none, not all."""
    good = MLSelector(model).select(cover).size
    total = cover.shape[0] * cover.shape[1]
    assert 0.01 * total < good < 0.99 * total
