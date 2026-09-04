"""
Round-trip tests: whatever goes in must come back out, unchanged.

The important one here is test_long_message.  The original project only
ever demonstrated a 45-character message, which is why nobody noticed
that anything longer fell apart.  So we deliberately fill most of the
image's capacity.
"""

import numpy as np
import pytest

from stego.core import CapacityError, capacity_bytes, hide, reveal
from stego.payload import CorruptMessageError, NoMessageError
from stego.selector import RandomSelector, SequentialSelector, VarianceSelector

SELECTORS = [SequentialSelector(), RandomSelector(), VarianceSelector()]


@pytest.fixture
def cover():
    """A 200x200 image with real texture, so VarianceSelector has work to do."""
    rng = np.random.default_rng(7)
    base = np.linspace(40, 220, 200, dtype=np.float64)
    img = np.stack([np.stack([base] * 200, axis=0)] * 3, axis=-1)
    img[100:, :, :] += rng.normal(0, 30, (100, 200, 3))     # textured half
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("selector", SELECTORS, ids=lambda s: s.name)
def test_short_message(cover, selector):
    message = "hello world"
    assert reveal(hide(cover, message, selector), selector) == message


@pytest.mark.parametrize("selector", SELECTORS, ids=lambda s: s.name)
def test_long_message(cover, selector):
    """Fill 90% of capacity -- the case the original version failed."""
    budget = capacity_bytes(cover, selector)
    message = "x" * int(budget * 0.9)
    assert reveal(hide(cover, message, selector), selector) == message


def test_empty_message(cover):
    selector = SequentialSelector()
    assert reveal(hide(cover, "", selector), selector) == ""


def test_cover_is_not_modified(cover):
    """hide() must return a copy, never edit the caller's array in place."""
    before = cover.copy()
    hide(cover, "hello", SequentialSelector())
    np.testing.assert_array_equal(cover, before)


def test_change_is_tiny(cover):
    """Every pixel moves by at most 1, and only in the blue channel."""
    stego = hide(cover, "hello world" * 20, SequentialSelector())
    difference = np.abs(stego.astype(int) - cover.astype(int))
    assert difference.max() <= 1
    assert difference[:, :, 0].sum() == 0        # red untouched
    assert difference[:, :, 1].sum() == 0        # green untouched


def test_message_too_big_raises(cover):
    selector = SequentialSelector()
    too_long = "x" * (capacity_bytes(cover, selector) + 1)
    with pytest.raises(CapacityError):
        hide(cover, too_long, selector)


def test_clean_image_reports_no_message(cover):
    """An ordinary photo must not decode to random junk."""
    with pytest.raises(NoMessageError):
        reveal(cover, SequentialSelector())


def test_wrong_selector_does_not_silently_lie(cover):
    """
    Decoding with the wrong selector must fail loudly, not return
    plausible-looking garbage.
    """
    stego = hide(cover, "the secret is 42", VarianceSelector())
    with pytest.raises((NoMessageError, CorruptMessageError)):
        reveal(stego, RandomSelector())
