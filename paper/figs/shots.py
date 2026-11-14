import sys
from pathlib import Path

from PIL import Image, ImageEnhance

src, here = Path(sys.argv[1]), Path(__file__).parent
for name, gain in (("heat0", 1.5), ("heat19", 1.5), ("flow", 1.5), ("split", 1.5), ("cars", 1.1)):
    f = src / f"{name}.png"
    if f.exists():
        im = Image.open(f).convert("RGB")
        w, h = im.size
        im = ImageEnhance.Brightness(im.crop((0, 60, w, h - 70))).enhance(gain)
        im.save(here / f"shot_{name}.jpg", quality=90)
        print(name)
