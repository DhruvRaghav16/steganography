"""
The command line interface.

Every command is a thin wrapper: load files, call the library, print the
result.  All the real logic lives in core.py, selector.py, training.py and
detect.py, so it can be tested without a terminal.

The README documents exactly these commands, and tests/test_cli.py runs
them, so the docs cannot drift out of date.
"""

import argparse
import glob
import pathlib
import sys

import joblib
import numpy as np
from PIL import Image

from . import core, detect, training
from .selector import MLSelector

#: Model files are NOT committed to the repo -- you build your own with
#: `stego train`.  Two reasons: joblib.load() on someone else's pickle is
#: arbitrary code execution, and a model is a build artifact, not source.
DEFAULT_MODEL = "model.pkl"

#: Lossless formats only.  See save_image().
SAFE_SUFFIXES = (".png", ".bmp")


def load_image(path):
    """Read an image file and return (H, W, 3) uint8 RGB."""
    return np.asarray(Image.open(path).convert("RGB"))


def save_image(img, path):
    """
    Write an image -- and refuse anything but PNG/BMP.

    JPEG is lossy: it transforms the image into the frequency domain and
    throws detail away to save space.  It does not preserve individual
    bits, so saving a stego image as JPEG destroys the hidden message
    completely and silently.  Failing loudly here is much kinder than
    handing back a file that looks fine and decodes to nothing.
    """
    if not str(path).lower().endswith(SAFE_SUFFIXES):
        raise SystemExit(
            f"refusing to write {path!r}: use .png\n"
            f"JPEG is lossy and would silently destroy the hidden message."
        )
    Image.fromarray(img).save(path)


def load_model(path):
    """Load the classifier, with a useful message if it is missing."""
    if not pathlib.Path(path).exists():
        raise SystemExit(
            f"no model at {path!r}\n"
            f"build one first:  python -m stego.cli train images/*.png"
        )
    return joblib.load(path)


def expand(patterns):
    """Expand shell globs ourselves -- Windows does not do it for us."""
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    return sorted(set(paths))


def cmd_train(args):
    paths = expand(args.images)
    if not paths:
        raise SystemExit("no images matched")

    print(f"training on {len(paths)} image(s):")
    for path in paths:
        print(f"  {path}")

    model, _stats = training.train(
        [load_image(p) for p in paths],
        n_estimators=args.estimators,
        seed=args.seed,
    )
    joblib.dump(model, args.out)
    print(f"saved {args.out}")


def cmd_hide(args):
    cover = load_image(args.image)
    selector = MLSelector(load_model(args.model))

    capacity = core.capacity_bytes(cover, selector)
    size = len(args.message.encode("utf-8"))

    stego = core.hide(cover, args.message, selector)
    save_image(stego, args.out)

    print(f"hid {size} bytes of {capacity} available ({size / capacity:.1%})")
    print(f"wrote {args.out}")


def cmd_reveal(args):
    selector = MLSelector(load_model(args.model))
    print(core.reveal(load_image(args.image), selector))


def cmd_capacity(args):
    cover = load_image(args.image)
    selector = MLSelector(load_model(args.model))
    print(f"{core.capacity_bytes(cover, selector)} bytes")


def cmd_analyze(args):
    result = detect.analyze(load_image(args.image))
    print(f"lsb correlation : {result['lsb_correlation']:.4f}  "
          f"(0.5 = random, higher = natural)")
    print(f"lsb entropy     : {result['lsb_entropy']:.4f}  "
          f"(near-useless on its own -- see detect.py)")
    print(f"chi-square p    : {result['chi_square_p']:.4f}")
    print(f"verdict         : {result['verdict']}")


def cmd_selftest(args):
    """Hide and reveal in one go -- checks the install actually works."""
    cover = load_image(args.image)
    selector = MLSelector(load_model(args.model))

    message = "selftest: नमस्ते / 汉字 / emoji 🙂"
    stego = core.hide(cover, message, selector)
    recovered = core.reveal(stego, selector)

    if recovered == message:
        print(f"round-trip OK  ({len(message.encode('utf-8'))} bytes, "
              f"unicode intact)")
    else:
        print(f"round-trip FAILED\n  sent: {message!r}\n  got : {recovered!r}")
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="stego",
        description="ML-assisted adaptive LSB steganography",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="fit a pixel-selection model")
    p.add_argument("images", nargs="+", help="cover images to learn from")
    p.add_argument("--out", default=DEFAULT_MODEL)
    p.add_argument("--estimators", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("hide", help="hide a message in an image")
    p.add_argument("--image", "-i", required=True)
    p.add_argument("--out", "-o", required=True, help="output PNG")
    p.add_argument("--message", "-m", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.set_defaults(fn=cmd_hide)

    p = sub.add_parser("reveal", help="extract a hidden message")
    p.add_argument("--image", "-i", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.set_defaults(fn=cmd_reveal)

    p = sub.add_parser("capacity", help="how much text fits in an image")
    p.add_argument("--image", "-i", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.set_defaults(fn=cmd_capacity)

    p = sub.add_parser("analyze", help="run steganalysis on an image")
    p.add_argument("--image", "-i", required=True)
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("selftest", help="hide+reveal once as an install check")
    p.add_argument("--image", "-i", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.set_defaults(fn=cmd_selftest)

    return parser


def force_utf8_output():
    """
    Make stdout/stderr UTF-8 capable.

    A Windows console defaults to a legacy code page (cp1252 here), which
    cannot represent Devanagari, CJK or emoji.  Without this, revealing a
    non-English message crashes in print() with UnicodeEncodeError -- the
    message was recovered perfectly, we just could not display it.  That
    is a miserable bug to debug, because the library is innocent.

    errors="replace" is deliberate: if a character truly cannot be shown,
    print a replacement glyph rather than throwing away the whole output.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:            # missing if stdout is piped oddly
            reconfigure(encoding="utf-8", errors="replace")


def main(argv=None):
    force_utf8_output()
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
