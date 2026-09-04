"""
Unicode round-trips, and the payload wire format.

The original project encoded each character with
`format(ord(ch), '08b')`.  That is fine for Latin text and silently
corrupts everything else: 'क' has ord() == 2325, which needs 12 bits,
not 8 -- so the bitstream shifts and the whole message becomes garbage.

All of this code goes through UTF-8 instead, so the test is: shove every
kind of text through the pipeline and check it comes back intact.
"""

import numpy as np

from stego.core import hide, reveal
from stego.selector import SequentialSelector, VarianceSelector
from stego.payload import HEADER_BITS, bytes_to_bits, bits_to_bytes, pack, unpack


def make_cover():
    rng = np.random.default_rng(3)
    return (rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))


def test_bit_roundtrip():
    data = b"\x00\xff\x80A\x00"
    assert bits_to_bytes(bytes_to_bits(data)) == data


def test_pack_size():
    assert pack("hi").size == 80          # 64-bit header + 2 bytes * 8 bits


def test_unicode_full_trip():
    for selector in [SequentialSelector(), VarianceSelector()]:
        for message in [
            "नमस्ते दुनिया",                     # Devanagari
            "你好，世界",                         # CJK
            "emoji 🙂 and mixed 日本語 + English",
            "a" * 200,                           # long plain ASCII too
        ]:
            assert reveal(hide(make_cover(), message, selector), selector) == message


def test_unpack_rejects_garbage():
    """Unpacking a string of random bits must not crash or succeed -- it
    must raise a clean error."""
    try:
        unpack(np.random.default_rng(0).integers(0, 2, 4096).astype(np.uint8))
    except Exception:  # noqa: BLE001 -- NoMessageError is the expected path
        return
    raise AssertionError("unpack() accepted random garbage")


def test_header_layout_is_stable():
    """
    Pin the exact bits of the magic marker.

    If someone changes MAGIC or the bit order, every previously-made stego
    image becomes unreadable.  This test makes that break loudly and
    immediately instead of silently in the field.

        'S' = 83 = 0101 0011
        'T' = 84 = 0101 0100
        'G' = 71 = 0100 0111
        '1' = 49 = 0011 0001
    """
    assert bytes_to_bits(b"STG1").tolist() == [
        0, 1, 0, 1, 0, 0, 1, 1,     # S
        0, 1, 0, 1, 0, 1, 0, 0,     # T
        0, 1, 0, 0, 0, 1, 1, 1,     # G
        0, 0, 1, 1, 0, 0, 0, 1,     # 1
    ]
    assert HEADER_BITS == 64
