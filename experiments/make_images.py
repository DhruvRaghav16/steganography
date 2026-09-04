"""
Write the synthetic cover images into images/.

The actual image generation lives in src/stego/synth.py so that the tests
use exactly the same pixels -- see that module for why the images look
the way they do (and why they deliberately have no global noise).

Run:  python experiments/make_images.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from stego.synth import LAYOUTS, make_cover

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "images"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for index, layout in enumerate(LAYOUTS, start=1):
        path = OUT_DIR / f"cover{index}_{layout}.png"
        Image.fromarray(make_cover(size=512, seed=index - 1, layout=layout)).save(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
