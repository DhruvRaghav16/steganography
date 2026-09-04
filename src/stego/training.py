"""
Build the training set BY MEASUREMENT, then fit the classifier.

WHY THIS FILE MATTERS
---------------------
The weakest part of most steganography projects is the training data: a
CSV of made-up labels with no explanation of where they came from.  A
model trained on invented labels cannot teach you anything -- it just
plays back the assumption you fed it.

So here the label is measured directly on real pixels, using a simplified
version of the HILL cost function from the steganography literature:

    residual = |Laplacian(gray)|        where is there high-frequency detail?
    local    = blur(residual, 3x3)      smooth it slightly
    spread   = blur(local, 15x15)       how much detail is AROUND this area?
    cost     = 1 / (spread + eps)       lots of detail nearby -> cheap to hide

    flat sky      little high-frequency energy  -> cost is HIGH -> avoid
    gravel/leaves lots of energy nearby         -> cost is LOW  -> use

The 15x15 window is the important part.  It means the label depends on a
much wider neighbourhood than the 3x3 features the classifier is given,
so the Random Forest is genuinely approximating something it cannot
compute directly from its inputs.  That is what makes this a real
learning task rather than a circular one.

A NOTE ON A COST FUNCTION THAT DOES NOT WORK
--------------------------------------------
An intuitive first attempt is "how much does flipping move the pixel away
from its neighbours?":

    cost = |flipped - local_mean| - |original - local_mean|

It looks reasonable and it is wrong.  Since flipped = original +- 1, that
expression evaluates to exactly +1 or -1 depending only on whether the
flip happens to move towards or away from the local mean.  It carries no
information about texture at all, and it splits any image almost 50/50.
Worth knowing, because the mistake is not visible until you measure.

Then the Random Forest learns to predict that cost from three cheap
features.  And we always print the score of a single hand-written
threshold next to it, so the ML has to earn its place.
"""

import numpy as np
from scipy.ndimage import laplace, uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .features import extract, stabilize, to_gray

#: Window used to pool the high-frequency residual.  Deliberately much
#: wider than the 3x3 the features use -- see the module docstring.
COST_WINDOW = 15

#: Keeps the reciprocal finite in perfectly flat regions.
COST_EPSILON = 1e-3

#: Pixels in the cheapest 40% become label 1, the dearest 40% become
#: label 0, and the middle 20% is thrown away.  Dropping the ambiguous
#: middle gives the model a cleaner signal to learn from.
LOW_QUANTILE = 0.40
HIGH_QUANTILE = 0.60

#: How many labelled pixels to keep per image.  A 512x512 image has
#: 260k of them; we do not need them all, and sampling keeps training fast.
SAMPLES_PER_IMAGE = 40_000

#: Label meaning "we dropped this pixel, do not train on it".
DROP = -1


def measure_cost(img):
    """
    Per-pixel embedding cost, HILL-style.  Lower = safer place to hide.

    Returns an (H, W) float array.

    Three steps:
      1. |Laplacian| -- where is the high-frequency detail?
      2. a small blur -- do not let a single noisy pixel decide.
      3. a wide blur -- is this pixel *surrounded* by detail?  A lone edge
         in an otherwise empty sky is still a bad place to hide, because
         an analyst looking at the region will notice.  Only somewhere
         with busy surroundings is genuinely safe.

    Then invert: lots of surrounding detail -> low cost -> good.
    """
    gray = to_gray(stabilize(img))

    residual = np.abs(laplace(gray))
    local = uniform_filter(residual, 3)
    spread = uniform_filter(local, COST_WINDOW)

    return 1.0 / (spread + COST_EPSILON)


def make_labels(cost):
    """Cost map -> flat array of 1 (good), 0 (bad), or DROP (ambiguous)."""
    flat = cost.ravel()
    low, high = np.quantile(flat, [LOW_QUANTILE, HIGH_QUANTILE])

    labels = np.full(flat.size, DROP, dtype=np.int8)
    labels[flat <= low] = 1                # cheap to embed here
    labels[flat >= high] = 0               # expensive, leave it alone
    return labels


def build_dataset(images, samples_per_image=SAMPLES_PER_IMAGE, seed=0):
    """List of images -> (X, y) ready for scikit-learn."""
    rng = np.random.default_rng(seed)
    feature_blocks, label_blocks = [], []

    for img in images:
        X = extract(img, stable=True)      # same features the selector uses
        y = make_labels(measure_cost(img))

        keep = np.flatnonzero(y != DROP)
        n = min(samples_per_image, keep.size)
        picked = rng.choice(keep, size=n, replace=False)

        feature_blocks.append(X[picked])
        label_blocks.append(y[picked])

    return np.vstack(feature_blocks), np.concatenate(label_blocks)


def best_variance_threshold(X_train, y_train, X_test, y_test):
    """
    The baseline the Random Forest has to beat.

    Tries a range of thresholds on the *training* half only, then scores
    the winner on the held-out half.  Picking the threshold on test data
    would be cheating, and would flatter the baseline.
    """
    variance_train = X_train[:, 1]
    candidates = np.quantile(variance_train, np.linspace(0.05, 0.95, 40))

    best_threshold, best_train_acc = candidates[0], -1.0
    for threshold in candidates:
        acc = ((variance_train > threshold).astype(int) == y_train).mean()
        if acc > best_train_acc:
            best_threshold, best_train_acc = threshold, acc

    test_acc = ((X_test[:, 1] > best_threshold).astype(int) == y_test).mean()
    return float(best_threshold), float(test_acc)


def train(images, n_estimators=100, seed=0, verbose=True, n_jobs=1):
    """
    Fit the Random Forest and return (model, stats_dict).

    The caller decides where to save it -- this function does no file I/O,
    which keeps it easy to test.

    n_jobs
        Threads for fitting.  Defaults to 1, not -1, on purpose.  Every
        later `model.predict(...)` inherits this value, and a forest
        predicting on a few thousand rows spends far longer starting
        worker threads than doing the arithmetic -- with n_jobs=-1 the
        test suite took 17 minutes instead of 3 seconds.  Pass -1 only
        for a genuinely large one-off fit.
    """
    X, y = build_dataset(images, seed=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=n_jobs,
        min_samples_leaf=5,        # a little smoothing; keeps the model small
    )
    model.fit(X_train, y_train)

    forest_acc = float(model.score(X_test, y_test))
    threshold, baseline_acc = best_variance_threshold(
        X_train, y_train, X_test, y_test
    )

    stats = {
        "n_samples": int(X.shape[0]),
        "test_accuracy": forest_acc,             # held-out, not training
        "baseline_threshold": threshold,
        "baseline_accuracy": baseline_acc,
        "gain_over_baseline": forest_acc - baseline_acc,
        "feature_importance": dict(
            zip(("brightness", "variance", "edge"), model.feature_importances_)
        ),
    }

    if verbose:
        print(f"  training samples      : {stats['n_samples']:,}")
        print(f"  baseline (variance>{threshold:.1f}) : {baseline_acc:.4f}")
        print(f"  random forest         : {forest_acc:.4f}")
        print(f"  gain from ML          : {stats['gain_over_baseline']:+.4f}")
        print("  feature importance    : ", end="")
        print(", ".join(f"{k}={v:.3f}" for k, v in
                        stats["feature_importance"].items()))

    return model, stats
