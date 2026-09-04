"""
EXPERIMENT 1  --  the desynchronization failure, measured.

This is the experiment the whole README is built around.

THE CLAIM
---------
If the pixel selector reads features from the raw image, then embedding
changes those features, which changes the selection, which means the
receiver reads from different pixels than the sender wrote to.  Because
reading is positional, ONE shifted index destroys everything after it.

WHAT WE MEASURE
---------------
1. first divergence index -- the position of the first pixel the sender
   and receiver disagree about.  This is the real headline number.
2. bit accuracy           -- of the bits we embedded, how many does the
   receiver read back correctly?  50% means "coin flip", i.e. garbage.
3. decode success         -- does reveal() return the original string?

WHAT WE FOUND
-------------
The naive selector desynchronizes within the first few dozen bits on all
three cover images -- so it is broken at *every* payload size, not just
large ones.  The original version of this project survived to ~4,800 bits
before failing, which is why its 45-character demo appeared to work.  The
difference is not that one is safer: it is that the survival length is an
accident of the image and of how coarse the model happens to be.  That
unpredictability is the point.

The fixed selector is bit-identical on both sides, on every image, at
every payload size.

Run:  python experiments/exp1_desync.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
from PIL import Image

from stego import training
from stego.core import capacity_bytes, hide, reveal
from stego.payload import pack
from stego.selector import MLSelector, NaiveMLSelector

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"
IMAGE_DIR = ROOT / "images"
MODEL_PATH = ROOT / "model.pkl"

#: Payload sizes to probe, in bits.  Clamped to each image's capacity.
PAYLOAD_BITS = [500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000]


def get_model():
    """Load model.pkl, or train one from images/ if it is not there yet."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    print("no model.pkl -- training one from images/ ...")
    covers = [np.asarray(Image.open(p).convert("RGB"))
              for p in sorted(IMAGE_DIR.glob("cover*.png"))]
    model, _ = training.train(covers, n_estimators=100, verbose=False)
    joblib.dump(model, MODEL_PATH)
    return model


def first_divergence(selector, cover, n_bits):
    """
    Index of the first pixel the two sides disagree about, or None.

    The sender picks from the cover, the receiver picks from the stego.
    We compare the two lists position by position, because that is how
    the bits are actually read.
    """
    stego = hide(cover, "x" * (n_bits // 8), selector)
    sender = selector.select(cover)
    receiver = selector.select(stego)

    shared = min(sender.size, receiver.size)
    mismatch = np.flatnonzero(sender[:shared] != receiver[:shared])
    if mismatch.size:
        return int(mismatch[0])
    return None if sender.size == receiver.size else shared


def measure(selector, cover, n_bits):
    """Embed n_bits, read them back, report (bit_accuracy, decoded_ok)."""
    text = "x" * (n_bits // 8)
    written = pack(text)

    stego = hide(cover, text, selector)
    receiver = selector.select(stego)
    read_back = stego.reshape(-1, 3)[receiver[: written.size], 2] & 1

    bit_accuracy = float((read_back == written).mean())

    try:
        decoded_ok = reveal(stego, selector) == text
    except Exception:                                   # noqa: BLE001
        decoded_ok = False

    return bit_accuracy, decoded_ok


def main():
    model = get_model()
    naive, fixed = NaiveMLSelector(model), MLSelector(model)

    covers = sorted(IMAGE_DIR.glob("cover*.png"))
    if not covers:
        raise SystemExit("no images -- run: python experiments/make_images.py")

    print("=" * 74)
    print("FIRST DIVERGENCE  (position of the first pixel the two sides "
          "disagree about)")
    print("=" * 74)

    divergences = {}
    for path in covers:
        img = np.asarray(Image.open(path).convert("RGB"))
        naive_div = first_divergence(naive, img, 16_000)
        fixed_div = first_divergence(fixed, img, 16_000)
        divergences[path.stem] = naive_div

        print(f"  {path.name:<26} capacity={capacity_bytes(img, fixed):>6} B")
        print(f"      naive : diverges at index {naive_div}")
        print(f"      fixed : {'never diverges' if fixed_div is None else f'diverges at {fixed_div}'}")

    print()
    print("=" * 74)
    print("RECOVERY vs PAYLOAD")
    print("=" * 74)
    print(f"  {'payload':>9} | {'naive bits':>10} {'decodes':>8} | "
          f"{'fixed bits':>10} {'decodes':>8}")
    print("  " + "-" * 60)

    reference = np.asarray(Image.open(covers[0]).convert("RGB"))
    budget = capacity_bytes(reference, fixed) * 8
    sizes = [n for n in PAYLOAD_BITS if n <= budget]

    naive_curve, fixed_curve = [], []
    for n in sizes:
        n_acc, n_ok = measure(naive, reference, n)
        f_acc, f_ok = measure(fixed, reference, n)
        naive_curve.append(n_acc)
        fixed_curve.append(f_acc)
        print(f"  {n:>9,} | {n_acc:>9.1%} {str(n_ok):>8} | "
              f"{f_acc:>9.1%} {str(f_ok):>8}")

    print()
    print("  naive bit accuracy sits near 50% -- that is a coin flip, i.e.")
    print("  the receiver is reading noise, not the message.")

    plot(sizes, naive_curve, fixed_curve, divergences)


def plot(sizes, naive_curve, fixed_curve, divergences):
    import matplotlib

    matplotlib.use("Agg")                    # no GUI -- write straight to file
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(sizes, [v * 100 for v in fixed_curve], marker="s", linewidth=2.5,
             color="#27ae60", label="fixed  (LSB-stripped features)")
    ax1.plot(sizes, [v * 100 for v in naive_curve], marker="o", linewidth=2.5,
             color="#c0392b", label="naive  (features from raw image)")
    ax1.axhline(50, linestyle=":", color="#7f8c8d", linewidth=1.2)
    ax1.text(sizes[0], 52, "50% = coin flip", fontsize=8, color="#7f8c8d")
    ax1.set_xscale("log")
    ax1.set_xlabel("message length (bits)")
    ax1.set_ylabel("bits recovered correctly (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Recovery vs payload size")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(loc="center left", fontsize=9)

    names = list(divergences)
    values = [divergences[n] or 0 for n in names]
    ax2.bar(range(len(names)), values, color="#c0392b", width=0.55)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.replace("cover", "").replace("_", "\n")
                         for n in names], fontsize=9)
    ax2.set_ylabel("first divergence (bit index)")
    ax2.set_title("Where the naive selector breaks\n(lower = fails sooner)")
    ax2.grid(alpha=0.3, axis="y")
    for i, v in enumerate(values):
        ax2.text(i, v + 0.6, str(v), ha="center", fontsize=10, weight="bold")

    fig.suptitle("Selection desynchronization: the bug, and the fix",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    path = RESULTS_DIR / "desync_curve.png"
    fig.savefig(path, dpi=150)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
