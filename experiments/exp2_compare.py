"""
EXPERIMENT 2  --  does the ML-guided selection actually help?

The central claim of a project like this is "choosing pixels with a
classifier is stealthier than choosing them naively."  A claim like that
is worth nothing without a baseline, so every selector here does the same
job with the same payload and the numbers go side by side -- including
the ones that make the ML look bad.

The four contenders
-------------------
sequential : top-left to bottom-right.  The textbook method.
random     : same pixels, shuffled.  Kills the "front of the image looks
             different" signature, but still ignores image content.
variance   : one hand-written rule -- keep pixels whose 3x3 variance is
             above a threshold.  This is the honest bar for the ML.  If a
             Random Forest cannot beat a single `if`, say so.
ml         : the trained Random Forest.

Two details that matter for fairness
------------------------------------
1. The payload is RANDOM text, not "xxxx...".  A repeated character is a
   repeating bit pattern, which leaves its own structure in the LSB plane
   and flatters the detector in a way real (compressed, encrypted) data
   never would.
2. Every selector carries the SAME number of bytes, capped by whichever
   selector has the least capacity.  Otherwise the ML would score well
   simply by hiding less.

What the columns mean
---------------------
capacity   how many bytes that selector could hold.  Adaptive selectors
           buy their stealth by using fewer pixels -- that is the trade.
PSNR       higher = less visible.  All methods clear 60 dB, because LSB
           embedding is a tiny change by construction.  PSNR measures
           VISIBILITY, not detectability -- do not read it as safety.
corr       LSB neighbour correlation.  THE detectability number.  The
           clean image sets the reference; the closer a stego image stays
           to it, the harder it is to spot.  0.5 = pure coin flip.
drop       clean_corr - stego_corr.  Lower is better.  This is the column
           that answers the question in the title.

Run:  python experiments/exp2_compare.py
"""

import csv
import pathlib
import random
import string
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
from PIL import Image

from stego.core import capacity_bytes, hide
from stego.detect import chi_square, lsb_correlation, lsb_entropy, psnr
from stego.selector import (
    MLSelector,
    RandomSelector,
    SequentialSelector,
    VarianceSelector,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"
IMAGE_DIR = ROOT / "images"
MODEL_PATH = ROOT / "model.pkl"

#: Fraction of the *smallest* selector's capacity to fill.
PAYLOAD_FRACTION = 0.5

#: Fixed seed so the payload is identical for every selector and every run.
PAYLOAD_SEED = 20240101


def random_payload(n_bytes, seed=PAYLOAD_SEED):
    """
    n_bytes of unpredictable printable text.

    Real hidden payloads are compressed or encrypted, so their bits look
    random.  Using "xxxx..." instead would embed a repeating pattern and
    quietly change what the detectors see.
    """
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return "".join(rng.choice(alphabet) for _ in range(n_bytes))


def build_row(name, selector, cover, text, clean_corr):
    stego = hide(cover, text, selector)
    corr = lsb_correlation(stego)
    return {
        "selector": name,
        "capacity_bytes": capacity_bytes(cover, selector),
        "psnr_db": psnr(cover, stego),
        "lsb_correlation": corr,
        "correlation_drop": clean_corr - corr,
        "lsb_entropy": lsb_entropy(stego),
        "chi2_p": chi_square(stego),
        "pixels_changed": int((stego[:, :, 2] != cover[:, :, 2]).sum()),
    }


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("no model.pkl -- run experiments/exp1_desync.py first")
    model = joblib.load(MODEL_PATH)

    selectors = [
        ("sequential", SequentialSelector()),
        ("random", RandomSelector()),
        ("variance", VarianceSelector()),
        ("ml", MLSelector(model)),
    ]

    covers = sorted(IMAGE_DIR.glob("cover*.png"))
    if not covers:
        raise SystemExit("no images -- run: python experiments/make_images.py")

    all_rows = []
    for path in covers:
        cover = np.asarray(Image.open(path).convert("RGB"))
        clean_corr = lsb_correlation(cover)

        smallest = min(capacity_bytes(cover, s) for _, s in selectors)
        payload_bytes = int(smallest * PAYLOAD_FRACTION)
        text = random_payload(payload_bytes)

        print(f"\n{path.name}   payload = {payload_bytes:,} random bytes")
        print(f"  {'selector':<11} {'capacity':>9} {'PSNR dB':>8} "
              f"{'corr':>7} {'drop':>7}")
        print("  " + "-" * 48)
        print(f"  {'(clean)':<11} {'-':>9} {'inf':>8} {clean_corr:>7.4f} "
              f"{'-':>7}")

        for name, selector in selectors:
            row = build_row(name, selector, cover, text, clean_corr)
            row["image"] = path.name
            all_rows.append(row)
            print(f"  {name:<11} {row['capacity_bytes']:>9,} "
                  f"{row['psnr_db']:>8.2f} {row['lsb_correlation']:>7.4f} "
                  f"{row['correlation_drop']:>7.4f}")

    write_csv(all_rows)
    summarize(all_rows)
    plot(all_rows)


def write_csv(rows):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "comparison.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {path}")


NAMES = ("sequential", "random", "variance", "ml")


def mean_of(rows, field, name):
    return float(np.mean([r[field] for r in rows if r["selector"] == name]))


def summarize(rows):
    print("\n" + "=" * 62)
    print("AVERAGE ACROSS ALL IMAGES   (lower drop = harder to detect)")
    print("=" * 62)
    print(f"  {'selector':<11} {'PSNR dB':>9} {'corr drop':>11} {'capacity':>10}")
    print("  " + "-" * 46)
    for name in NAMES:
        print(f"  {name:<11} {mean_of(rows, 'psnr_db', name):>9.2f} "
              f"{mean_of(rows, 'correlation_drop', name):>11.4f} "
              f"{mean_of(rows, 'capacity_bytes', name):>10,.0f}")

    drops = {n: mean_of(rows, "correlation_drop", n) for n in NAMES}
    best = min(drops, key=drops.get)
    print(f"\n  hardest to detect : {best}  (drop {drops[best]:.4f})")

    ml_gain = drops["sequential"] - drops["ml"]
    vs_variance = drops["variance"] - drops["ml"]
    print(f"  ml vs sequential  : {ml_gain:+.4f}  "
          f"({'better' if ml_gain > 0 else 'worse'})")
    print(f"  ml vs variance    : {vs_variance:+.4f}  "
          f"({'better' if vs_variance > 0 else 'worse'})")

    print()
    if abs(vs_variance) < 0.005:
        print("  VERDICT: the Random Forest is level with a one-line variance")
        print("  threshold.  The ML is not what makes this project work -- the")
        print("  synchronization fix is.  Reporting that plainly is worth more")
        print("  than a tuned number.")
    elif vs_variance > 0:
        print("  VERDICT: the Random Forest genuinely beats the hand-written")
        print("  rule.  The ML earns its place.")
    else:
        print("  VERDICT: the hand-written threshold beats the Random Forest.")
        print("  The ML does not earn its place on these images.")


def plot(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#95a5a6", "#95a5a6", "#3498db", "#27ae60"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    ax1.bar(NAMES, [mean_of(rows, "correlation_drop", n) for n in NAMES],
            color=colours)
    ax1.set_ylabel("LSB correlation drop")
    ax1.set_title("Detectability  (lower = better)")
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(NAMES, [mean_of(rows, "capacity_bytes", n) for n in NAMES],
            color=colours)
    ax2.set_ylabel("capacity (bytes)")
    ax2.set_title("Capacity  (higher = more room)")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("The trade adaptive selection makes: less capacity, "
                 "less detectable", weight="bold")
    fig.tight_layout()
    path = RESULTS_DIR / "comparison.png"
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
