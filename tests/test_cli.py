"""
CLI tests.

These exist for one specific reason: the original version of this project
documented a `steg_cli.py` with encode/decode subcommands that was never
committed.  Every install instruction in its README failed on the first
command.

So every command shown in our README is executed here.  If someone
renames a flag, the docs break loudly in CI instead of quietly for the
next person who clones the repo.
"""

import pathlib
import sys

import joblib
import numpy as np
import pytest
from PIL import Image

from stego.cli import main
from stego.synth import LAYOUTS, make_cover
from stego.training import train

SIZE = 128


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A directory with three cover PNGs and a trained model in it."""
    root = tmp_path_factory.mktemp("cli")

    covers = []
    for index, layout in enumerate(LAYOUTS, start=1):
        img = make_cover(SIZE, seed=index - 1, layout=layout)
        path = root / f"cover{index}_{layout}.png"
        Image.fromarray(img).save(path)
        covers.append(img)

    model, _stats = train(covers, n_estimators=12, seed=0, verbose=False)
    joblib.dump(model, root / "model.pkl")

    return root


def run(workspace, *argv):
    """Invoke the CLI the way a user would, with paths inside workspace."""
    main([str(a) for a in argv])


def test_hide_then_reveal(workspace, capsys):
    cover = workspace / "cover1_landscape.png"
    stego = workspace / "stego.png"
    model = workspace / "model.pkl"
    message = "meeting at 7pm"

    run(workspace, "hide", "-i", cover, "-o", stego, "-m", message,
        "--model", model)
    capsys.readouterr()

    run(workspace, "reveal", "-i", stego, "--model", model)
    assert capsys.readouterr().out.strip() == message


def test_reveal_handles_unicode(workspace, capsys):
    """The whole point of using UTF-8 instead of ord()."""
    cover = workspace / "cover2_inverted.png"
    stego = workspace / "unicode.png"
    model = workspace / "model.pkl"
    message = "नमस्ते / 汉字 / 🙂"

    run(workspace, "hide", "-i", cover, "-o", stego, "-m", message,
        "--model", model)
    capsys.readouterr()

    run(workspace, "reveal", "-i", stego, "--model", model)
    assert capsys.readouterr().out.strip() == message


def test_capacity_reports_bytes(workspace, capsys):
    run(workspace, "capacity", "-i", workspace / "cover1_landscape.png",
        "--model", workspace / "model.pkl")
    out = capsys.readouterr().out
    assert "bytes" in out
    assert int(out.split()[0]) > 0


def test_analyze_runs_on_clean_image(workspace, capsys):
    run(workspace, "analyze", "-i", workspace / "cover1_landscape.png")
    out = capsys.readouterr().out
    assert "lsb correlation" in out
    assert "verdict" in out


def test_selftest_passes(workspace, capsys):
    run(workspace, "selftest", "-i", workspace / "cover3_patchwork.png",
        "--model", workspace / "model.pkl")
    assert "OK" in capsys.readouterr().out


def test_refuses_jpeg_output(workspace):
    """JPEG would destroy the message -- fail loudly, not silently."""
    with pytest.raises(SystemExit) as exc:
        run(workspace, "hide", "-i", workspace / "cover1_landscape.png",
            "-o", workspace / "bad.jpg", "-m", "hi",
            "--model", workspace / "model.pkl")
    assert "png" in str(exc.value).lower()


def test_missing_model_gives_useful_error(workspace):
    with pytest.raises(SystemExit) as exc:
        run(workspace, "reveal", "-i", workspace / "cover1_landscape.png",
            "--model", workspace / "does_not_exist.pkl")
    assert "train" in str(exc.value)


def test_train_writes_a_model(workspace, tmp_path):
    out = tmp_path / "fresh.pkl"
    run(workspace, "train", workspace / "cover1_landscape.png",
        "--out", out, "--estimators", 8)
    assert out.exists()
