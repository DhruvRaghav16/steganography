"""
The wire format: how a string becomes a stream of bits.

Everything the receiver needs is in the header, so it never has to guess
where the message stops.

    bit 0            bit 32              bit 64
    |                |                   |
    +----------------+-------------------+---------------------------+
    | magic  "STG1"  | length  (4 bytes) | text, UTF-8 encoded       |
    | 32 bits        | 32 bits, big-end  | length * 8 bits           |
    +----------------+-------------------+---------------------------+

Why each part exists
--------------------
magic  : lets us say "there is nothing hidden here" instead of returning
         garbage for an ordinary photo.  Also carries a version number,
         so a future STG2 format can coexist.
length : tells the receiver exactly when to stop.  The alternative -- a
         terminator byte -- breaks the moment that byte appears inside
         the message itself.
text   : always UTF-8 encoded.  The original version used
         `format(ord(ch), '08b')`, which silently produces 15 bits for a
         character like 'क' and corrupts everything after it.

The header sits at bit 0 and we read it there.  We never *search* the
stream for the magic bytes -- we can trust the position because
features.stabilize() guarantees both sides read the same pixels in the
same order.
"""

import numpy as np

#: Format marker + version.  4 bytes.
MAGIC = b"STG1"

#: Bytes used to store the payload length.  4 bytes -> up to 4 GB.
LENGTH_BYTES = 4

#: Total header size in bits: (4 + 4) * 8 = 64.
HEADER_BITS = (len(MAGIC) + LENGTH_BYTES) * 8


class NoMessageError(Exception):
    """No valid header found -- this image probably carries nothing."""


class CorruptMessageError(Exception):
    """Header looked valid but the payload did not survive intact."""


def bytes_to_bits(data):
    """b'A' -> array([0,1,0,0,0,0,0,1], uint8).  MSB first."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits):
    """The exact inverse of bytes_to_bits.  Length must be a multiple of 8."""
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def pack(text):
    """
    str -> uint8 array of 0/1 bits, ready to be written into pixels.

    >>> pack("hi").size
    80
    """
    body = text.encode("utf-8")
    header = MAGIC + len(body).to_bytes(LENGTH_BYTES, "big")
    return bytes_to_bits(header + body)


def unpack(bits):
    """
    uint8 array of 0/1 bits -> str.  The exact inverse of pack().

    Raises NoMessageError if the magic bytes are absent, and
    CorruptMessageError if the header is fine but the body is not.
    """
    bits = np.asarray(bits, dtype=np.uint8)

    if bits.size < HEADER_BITS:
        raise NoMessageError("image is too small to even hold a header")

    header = bits_to_bytes(bits[:HEADER_BITS])
    if header[: len(MAGIC)] != MAGIC:
        raise NoMessageError("no hidden message found (magic bytes missing)")

    n_bytes = int.from_bytes(header[len(MAGIC) :], "big")
    end = HEADER_BITS + n_bytes * 8

    if bits.size < end:
        raise CorruptMessageError(
            f"header claims {n_bytes} bytes but only "
            f"{(bits.size - HEADER_BITS) // 8} are readable"
        )

    try:
        return bits_to_bytes(bits[HEADER_BITS:end]).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CorruptMessageError("payload is not valid UTF-8") from exc
