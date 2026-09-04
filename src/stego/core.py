"""
hide() and reveal() -- the heart of the project.

The two functions are mirror images of each other:

    hide  : text -> bits -> write into the LSB of chosen pixels
    reveal:         read the LSB of the same pixels -> bits -> text

"the same pixels" is the whole trick, and it is guaranteed by
features.stabilize().  See selector.py for why that matters.
"""

import numpy as np

from .features import BLUE
from .payload import HEADER_BITS, pack, unpack


class CapacityError(Exception):
    """The message needs more pixels than this selector offers."""


def capacity_bytes(img, selector):
    """
    How many bytes of text this image can hold with this selector.

    We subtract the 64-bit header first, because those bits are overhead
    and not available to the user's message.
    """
    usable = selector.capacity_bits(img) - HEADER_BITS
    return max(usable // 8, 0)


def hide(img, text, selector):
    """
    Return a copy of `img` with `text` hidden inside it.

    Parameters
    ----------
    img      : (H, W, 3) uint8 RGB array.
    text     : the message.  Any Unicode -- it is UTF-8 encoded.
    selector : decides which pixels are used, and in what order.

    Raises CapacityError if the message does not fit.
    """
    bits = pack(text)
    idx = selector.select(img)

    if bits.size > idx.size:
        raise CapacityError(
            f"message needs {bits.size} bits but this image only offers "
            f"{idx.size} ({capacity_bytes(img, selector)} bytes of text). "
            f"Use a bigger image or a shorter message."
        )

    out = img.copy()
    flat = out.reshape(-1, 3)              # a view, so writes hit `out`
    target = idx[: bits.size]              # only the pixels we actually need

    # Clear the last bit, then OR in our bit.
    #   0xFE = 11111110  -> `& 0xFE` wipes the LSB, keeps the other 7
    #   `| bit`          -> writes the new LSB
    # The pixel value moves by at most 1 out of 255, which is invisible.
    flat[target, BLUE] = (flat[target, BLUE] & 0xFE) | bits.astype(np.uint8)

    return out


def reveal(img, selector):
    """
    Pull the hidden text back out.  Needs only the stego image and the
    same selector -- never the original cover image.  That is what makes
    the scheme "blind".

    Raises payload.NoMessageError if nothing is hidden here, and
    payload.CorruptMessageError if the header survived but the body did not.
    """
    idx = selector.select(img)                       # same list as hide()
    bits = img.reshape(-1, 3)[idx, BLUE] & 1         # `& 1` keeps just the LSB
    return unpack(bits)


def recovery_rate(original, decoded):
    """
    What fraction of the message came back correctly, position by position.

    Used by exp1_desync.py to draw the failure curve.  Positional
    comparison is deliberate: a one-bit shift ruins everything after it,
    and this number is meant to show that.
    """
    if not original:
        return 1.0
    if decoded is None:
        return 0.0
    matched = sum(a == b for a, b in zip(original, decoded))
    return matched / len(original)
